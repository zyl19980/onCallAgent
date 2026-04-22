create table if not exists documents (
    id                bigserial primary key,
    collection_name   varchar(128) not null,
    source_path       text not null,
    file_name         varchar(512) not null,
    file_ext          varchar(32) not null,
    file_hash         varchar(128) not null,
    status            varchar(32) not null default 'active',
    metadata          jsonb not null default '{}'::jsonb,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),
    unique (collection_name, source_path)
);

create index if not exists idx_documents_source_path on documents(source_path);
create index if not exists idx_documents_status on documents(status);

create table if not exists document_chunks (
    id                  bigserial primary key,
    chunk_key           varchar(1024) not null unique,
    document_id         bigint not null references documents(id) on delete cascade,
    chunk_index         integer not null,
    source_text         text not null,
    published_text      text not null,
    draft_text          text,
    page_number         integer,
    section_path        text,
    chunk_type          varchar(64),
    metadata            jsonb not null default '{}'::jsonb,
    published_version   integer not null default 1,
    sync_status         varchar(32) not null default 'synced',
    last_publish_error  text,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now(),
    unique (document_id, chunk_index)
);

create index if not exists idx_document_chunks_document_id on document_chunks(document_id);
create index if not exists idx_document_chunks_sync_status on document_chunks(sync_status);

create table if not exists low_confidence_events (
    id                  bigserial primary key,
    session_id          varchar(128),
    user_id             varchar(128),
    raw_query           text not null,
    normalized_query    text not null,
    query_fingerprint   varchar(256) not null,
    reason              text,
    overall_confidence  varchar(16) not null,
    top1_score          numeric(8,4),
    top2_score          numeric(8,4),
    avg_top3_score      numeric(8,4),
    query_analysis      jsonb not null default '{}'::jsonb,
    retrieval_debug     jsonb not null default '{}'::jsonb,
    created_at          timestamptz not null default now()
);

create index if not exists idx_low_conf_events_fingerprint on low_confidence_events(query_fingerprint);
create index if not exists idx_low_conf_events_created_at on low_confidence_events(created_at);

create table if not exists low_confidence_event_chunks (
    id                    bigserial primary key,
    event_id              bigint not null references low_confidence_events(id) on delete cascade,
    chunk_id              bigint references document_chunks(id) on delete set null,
    chunk_key_snapshot    varchar(1024) not null,
    chunk_text_snapshot   text not null,
    file_name_snapshot    varchar(512),
    page_number_snapshot  integer,
    section_path_snapshot text,
    rank_no               integer not null,
    vector_score          numeric(8,4),
    keyword_score         numeric(8,4),
    fused_score           numeric(8,4),
    rerank_score          numeric(8,4),
    document_confidence   varchar(16) not null,
    matched_queries       jsonb not null default '[]'::jsonb,
    created_at            timestamptz not null default now()
);

create index if not exists idx_low_conf_event_chunks_event_id on low_confidence_event_chunks(event_id);
create index if not exists idx_low_conf_event_chunks_chunk_id on low_confidence_event_chunks(chunk_id);
create index if not exists idx_low_conf_event_chunks_chunk_key on low_confidence_event_chunks(chunk_key_snapshot);

create table if not exists chunk_edit_history (
    id                bigserial primary key,
    chunk_id          bigint not null references document_chunks(id) on delete cascade,
    version_no        integer not null,
    old_text          text not null,
    new_text          text not null,
    editor            varchar(128) not null,
    edit_note         text,
    publish_status    varchar(32) not null default 'published',
    created_at        timestamptz not null default now()
);

create index if not exists idx_chunk_edit_history_chunk_id on chunk_edit_history(chunk_id);
