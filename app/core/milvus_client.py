"""Milvus 客户端工厂模块。"""

from __future__ import annotations

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
    connections,
    utility,
)

from app.config import config


def _patch_pymilvus_milvus_client_orm_alias() -> None:
    if getattr(_patch_pymilvus_milvus_client_orm_alias, "_done", False):
        return
    try:
        from pymilvus.milvus_client.milvus_client import MilvusClient as InternalMilvusClient
    except ImportError:
        return

    original_init = InternalMilvusClient.__init__

    def wrapped_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        self._using = "default"

    InternalMilvusClient.__init__ = wrapped_init  # type: ignore[method-assign]
    setattr(_patch_pymilvus_milvus_client_orm_alias, "_done", True)


class MilvusClientManager:
    """Milvus 客户端管理器，支持多 collection。"""

    VECTOR_DIM: int = 1024
    ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH: int = 8000
    DEFAULT_SHARD_NUMBER: int = 2

    def __init__(self) -> None:
        self._client: MilvusClient | None = None
        self._collections: dict[str, Collection] = {}

    def connect(self) -> MilvusClient:
        if self._client is not None:
            logger.debug("Milvus 已连接，跳过重复 connect")
            return self._client

        try:
            _patch_pymilvus_milvus_client_orm_alias()
            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,
            )

            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)
            logger.info("成功连接到 Milvus")
            return self._client

        except MilvusException as exc:
            logger.error(f"Milvus 操作失败: {exc}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {exc}") from exc
        except Exception as exc:
            logger.error(f"连接 Milvus 失败: {exc}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {exc}") from exc

    def ensure_collection(self, collection_name: str) -> Collection:
        self.connect()

        if collection_name in self._collections:
            return self._collections[collection_name]

        if not self._collection_exists(collection_name):
            logger.info(f"collection '{collection_name}' 不存在，正在创建...")
            collection = self._create_collection(collection_name)
            logger.info(f"成功创建 collection '{collection_name}'")
        else:
            logger.info(f"collection '{collection_name}' 已存在")
            collection = Collection(collection_name)
            self._validate_vector_dim(collection_name, collection)

        self._load_collection(collection_name, collection)
        self._collections[collection_name] = collection
        return collection

    def get_collection(self, collection_name: str | None = None) -> Collection:
        target = collection_name or config.rag_collection_name
        return self.ensure_collection(target)

    def health_check(self) -> bool:
        try:
            if self._client is None:
                return False
            _ = connections.list_connections()
            return True
        except Exception as exc:
            logger.error(f"Milvus 健康检查失败: {exc}")
            return False

    def close(self) -> None:
        errors = []
        for name, collection in list(self._collections.items()):
            try:
                collection.release()
            except Exception as exc:
                errors.append(f"释放 collection {name} 失败: {exc}")
        self._collections.clear()

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as exc:
            errors.append(f"断开连接失败: {exc}")

        self._client = None

        if errors:
            logger.error(f"关闭 Milvus 连接时出现错误: {'; '.join(errors)}")
        else:
            logger.info("已关闭 Milvus 连接")

    def _collection_exists(self, collection_name: str) -> bool:
        return bool(utility.has_collection(collection_name))  # type: ignore[arg-type]

    def _create_collection(self, collection_name: str) -> Collection:
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=f"Knowledge collection: {collection_name}",
            enable_dynamic_field=False,
        )

        collection = Collection(
            name=collection_name,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,
        )
        self._create_index(collection)
        return collection

    def _create_index(self, collection: Collection) -> None:
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        _ = collection.create_index(field_name="vector", index_params=index_params)
        logger.info(f"成功为 collection '{collection.name}' 的 vector 字段创建索引")

    def _load_collection(self, collection_name: str, collection: Collection) -> None:
        try:
            load_state = utility.load_state(collection_name)
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                collection.load()
                logger.info(f"成功加载 collection '{collection_name}'")
            else:
                logger.info(f"Collection '{collection_name}' 已加载")
        except AttributeError:
            try:
                collection.load()
                logger.info(f"成功加载 collection '{collection_name}'")
            except MilvusException as exc:
                error_msg = str(exc).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{collection_name}' 已加载")
                else:
                    raise

    def _validate_vector_dim(self, collection_name: str, collection: Collection) -> None:
        schema = collection.schema
        for field in schema.fields:
            if field.name != "vector":
                continue
            params = getattr(field, "params", {})
            existing_dim = params.get("dim")
            if existing_dim is not None and existing_dim != self.VECTOR_DIM:
                raise RuntimeError(
                    f"collection '{collection_name}' 的向量维度为 {existing_dim}，与当前配置 {self.VECTOR_DIM} 不一致"
                )
            return

    def __enter__(self) -> "MilvusClientManager":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()


milvus_manager = MilvusClientManager()
