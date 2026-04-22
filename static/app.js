// SuperBizAgent 前端应用
class SuperBizAgentApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:9900/api';
        this.currentMode = 'quick'; // 'quick' 或 'stream'
        this.sessionId = this.generateSessionId();
        this.isStreaming = false;
        this.currentChatHistory = []; // 当前对话的消息历史
        this.chatHistories = this.loadChatHistories(); // 所有历史对话
        this.isCurrentChatFromHistory = false; // 标记当前对话是否是从历史记录加载的
        this.currentMainView = 'chat';
        this.fingerprintList = [];
        this.currentFingerprint = null;
        this.currentFingerprintDetail = null;
        this.currentChunk = null;
        this.currentChunkHistory = [];
        this.isGovernanceLoading = false;
        this.isChunkPublishing = false;
        this.isDraftDirty = false;
        
        this.initializeElements();
        this.bindEvents();
        this.updateUI();
        this.initMarkdown();
        this.checkAndSetCentered();
        this.renderChatHistory();
        this.switchMainView('chat');
    }

    // 初始化Markdown配置
    initMarkdown() {
        // 等待 marked 库加载完成
        const checkMarked = () => {
            if (typeof marked !== 'undefined') {
                try {
                    // 配置marked选项
                    marked.setOptions({
                        breaks: true,  // 支持GFM换行
                        gfm: true,     // 启用GitHub风格的Markdown
                        headerIds: false,
                        mangle: false
                    });

                    // 配置代码高亮
                    if (typeof hljs !== 'undefined') {
                        marked.setOptions({
                            highlight: function(code, lang) {
                                if (lang && hljs.getLanguage(lang)) {
                                    try {
                                        return hljs.highlight(code, { language: lang }).value;
                                    } catch (err) {
                                        console.error('代码高亮失败:', err);
                                    }
                                }
                                return code;
                            }
                        });
                    }
                    console.log('Markdown 渲染库初始化成功');
                } catch (e) {
                    console.error('Markdown 配置失败:', e);
                }
            } else {
                // 如果 marked 还没加载，等待一段时间后重试
                setTimeout(checkMarked, 100);
            }
        };
        checkMarked();
    }

    // 安全地渲染 Markdown
    renderMarkdown(content) {
        if (!content) return '';
        
        // 检查 marked 是否可用
        if (typeof marked === 'undefined') {
            console.warn('marked 库未加载，使用纯文本显示');
            return this.escapeHtml(content);
        }
        
        try {
            const html = marked.parse(content);
            return html;
        } catch (e) {
            console.error('Markdown 渲染失败:', e);
            return this.escapeHtml(content);
        }
    }

    // 高亮代码块
    highlightCodeBlocks(container) {
        if (typeof hljs !== 'undefined' && container) {
            try {
                container.querySelectorAll('pre code').forEach((block) => {
                    if (!block.classList.contains('hljs')) {
                        hljs.highlightElement(block);
                    }
                });
            } catch (e) {
                console.error('代码高亮失败:', e);
            }
        }
    }

    // 初始化DOM元素
    initializeElements() {
        // 侧边栏元素
        this.sidebar = document.querySelector('.sidebar');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.governanceEntryBtn = document.getElementById('governanceEntryBtn');
        this.aiOpsSidebarBtn = document.getElementById('aiOpsSidebarBtn');
        
        // 输入区域元素
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.toolsBtn = document.getElementById('toolsBtn');
        this.toolsMenu = document.getElementById('toolsMenu');
        this.uploadFileItem = document.getElementById('uploadFileItem');
        this.modeSelectorBtn = document.getElementById('modeSelectorBtn');
        this.modeDropdown = document.getElementById('modeDropdown');
        this.currentModeText = document.getElementById('currentModeText');
        this.fileInput = document.getElementById('fileInput');
        
        // 聊天区域元素
        this.chatMessages = document.getElementById('chatMessages');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.chatContainer = document.querySelector('.chat-container');
        this.welcomeGreeting = document.getElementById('welcomeGreeting');
        this.chatHistoryList = document.getElementById('chatHistoryList');
        this.governanceView = document.getElementById('governanceView');
        this.governanceRefreshBtn = document.getElementById('governanceRefreshBtn');
        this.governanceFingerprintList = document.getElementById('governanceFingerprintList');
        this.governanceDetail = document.getElementById('governanceDetail');
        this.governanceDetailHint = document.getElementById('governanceDetailHint');
        this.chunkEditorPanel = document.getElementById('chunkEditorPanel');
        this.chunkEditorHint = document.getElementById('chunkEditorHint');
        this.chunkEditorContent = document.getElementById('chunkEditorContent');
        
        // 初始化时检查是否需要居中
        this.checkAndSetCentered();
    }

    // 绑定事件监听器
    bindEvents() {
        // 新建对话
        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => {
                this.switchMainView('chat');
                this.newChat();
            });
        }

        if (this.governanceEntryBtn) {
            this.governanceEntryBtn.addEventListener('click', () => this.switchMainView('governance'));
        }
        
        // AI Ops按钮
        if (this.aiOpsSidebarBtn) {
            this.aiOpsSidebarBtn.addEventListener('click', () => this.triggerAIOps());
        }
        
        // 模式选择下拉菜单
        if (this.modeSelectorBtn) {
            this.modeSelectorBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleModeDropdown();
            });
        }
        
        // 下拉菜单项点击
        const dropdownItems = document.querySelectorAll('.dropdown-item');
        dropdownItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const mode = item.getAttribute('data-mode');
                this.selectMode(mode);
                this.closeModeDropdown();
            });
        });
        
        // 点击外部关闭下拉菜单
        document.addEventListener('click', (e) => {
            if (!this.modeSelectorBtn.contains(e.target) && 
                !this.modeDropdown.contains(e.target)) {
                this.closeModeDropdown();
            }
        });
        
        // 发送消息
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => this.sendMessage());
        }
        
        if (this.messageInput) {
            this.messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }
        
        // 工具按钮和菜单
        if (this.toolsBtn) {
            this.toolsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleToolsMenu();
            });
        }
        
        // 工具菜单项点击事件
        if (this.uploadFileItem) {
            this.uploadFileItem.addEventListener('click', () => {
                if (this.fileInput) {
                    this.fileInput.click();
                }
                this.closeToolsMenu();
            });
        }
        
        // 点击外部关闭工具菜单
        document.addEventListener('click', (e) => {
            if (this.toolsBtn && this.toolsMenu && 
                !this.toolsBtn.contains(e.target) && 
                !this.toolsMenu.contains(e.target)) {
                this.closeToolsMenu();
            }
        });
        
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        if (this.governanceRefreshBtn) {
            this.governanceRefreshBtn.addEventListener('click', () => this.loadFingerprintList({ force: true, withOverlay: true }));
        }

        if (this.governanceFingerprintList) {
            this.governanceFingerprintList.addEventListener('click', (event) => {
                const item = event.target.closest('[data-fingerprint]');
                if (item) {
                    this.openFingerprint(item.dataset.fingerprint);
                }
            });
        }

        if (this.governanceDetail) {
            this.governanceDetail.addEventListener('click', (event) => {
                const chunkCard = event.target.closest('[data-chunk-key]');
                if (chunkCard) {
                    this.openChunkEditor(chunkCard.dataset.chunkKey);
                }
            });
        }

        if (this.chunkEditorContent) {
            this.chunkEditorContent.addEventListener('input', (event) => {
                if (event.target && event.target.id === 'chunkDraftTextarea') {
                    this.isDraftDirty = true;
                    this.updateChunkEditorActions();
                }
            });

            this.chunkEditorContent.addEventListener('click', (event) => {
                const action = event.target.closest('[data-action]');
                if (!action) {
                    return;
                }

                const { action: actionName } = action.dataset;
                if (actionName === 'save-draft' && this.currentChunk) {
                    this.saveChunkDraft(this.currentChunk.chunk_key);
                } else if (actionName === 'publish-chunk' && this.currentChunk) {
                    this.publishChunk(this.currentChunk.chunk_key);
                } else if (actionName === 'view-history' && this.currentChunk) {
                    this.loadChunkHistory(this.currentChunk.chunk_key, { force: true });
                }
            });
        }
    }

    // 切换工具菜单显示/隐藏
    toggleToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) {
                wrapper.classList.toggle('active');
            }
        }
    }

    // 关闭工具菜单
    closeToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) {
                wrapper.classList.remove('active');
            }
        }
    }

    // 新建对话
    newChat() {
        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成后再新建对话', 'warning');
            return;
        }
        
        // 如果当前有对话内容，且不是从历史记录加载的，才保存为新的历史对话
        // 如果是从历史记录加载的，只需要更新该历史记录
        if (this.currentChatHistory.length > 0) {
            if (this.isCurrentChatFromHistory) {
                // 当前对话是从历史记录加载的，更新该历史记录
                this.updateCurrentChatHistory();
            } else {
                // 当前对话是新对话，保存为新的历史对话
                this.saveCurrentChat();
            }
        }
        
        // 停止所有进行中的操作
        this.isStreaming = false;
        
        // 清空输入框
        if (this.messageInput) {
            this.messageInput.value = '';
        }
        
        // 清空当前对话历史
        this.currentChatHistory = [];
        
        // 重置标记
        this.isCurrentChatFromHistory = false;
        
        // 清空聊天记录
        if (this.chatMessages) {
            this.chatMessages.innerHTML = '';
        }
        
        // 生成新的会话ID
        this.sessionId = this.generateSessionId();
        
        // 重置模式为快速
        this.currentMode = 'quick';
        this.updateUI();
        
        // 重新设置居中样式（确保对话框居中显示）
        this.checkAndSetCentered();
        
        // 确保容器有过渡动画
        if (this.chatContainer) {
            this.chatContainer.style.transition = 'all 0.5s ease';
        }
        
        // 更新历史对话列表
        this.renderChatHistory();
    }
    
    // 保存当前对话到历史记录（新建）
    saveCurrentChat() {
        if (this.currentChatHistory.length === 0) {
            return;
        }
        
        // 检查是否已存在相同ID的历史记录
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex !== -1) {
            // 如果已存在，更新而不是新建
            this.updateCurrentChatHistory();
            return;
        }
        
        // 获取对话标题（使用第一条用户消息的前30个字符）
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        const title = firstUserMessage ? 
            (firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '')) : 
            '新对话';
        
        const chatHistory = {
            id: this.sessionId,
            title: title,
            messages: [...this.currentChatHistory],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        
        // 添加到历史记录列表的开头
        this.chatHistories.unshift(chatHistory);
        
        // 限制历史记录数量（最多保存50条）
        if (this.chatHistories.length > 50) {
            this.chatHistories = this.chatHistories.slice(0, 50);
        }
        
        // 保存到localStorage
        this.saveChatHistories();
    }
    
    // 更新当前对话的历史记录
    updateCurrentChatHistory() {
        if (this.currentChatHistory.length === 0) {
            return;
        }
        
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex === -1) {
            // 如果不存在，调用保存方法
            this.saveCurrentChat();
            return;
        }
        
        // 更新现有的历史记录
        const history = this.chatHistories[existingIndex];
        history.messages = [...this.currentChatHistory];
        history.updatedAt = new Date().toISOString();
        
        // 如果标题需要更新（第一条消息改变了）
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        if (firstUserMessage) {
            const newTitle = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
            if (history.title !== newTitle) {
                history.title = newTitle;
            }
        }
        
        // 保存到localStorage
        this.saveChatHistories();
    }
    
    // 加载历史对话列表
    loadChatHistories() {
        try {
            const stored = localStorage.getItem('chatHistories');
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.error('加载历史对话失败:', e);
            return [];
        }
    }
    
    // 保存历史对话列表到localStorage
    saveChatHistories() {
        try {
            localStorage.setItem('chatHistories', JSON.stringify(this.chatHistories));
        } catch (e) {
            console.error('保存历史对话失败:', e);
        }
    }
    
    // 渲染历史对话列表
    renderChatHistory() {
        if (!this.chatHistoryList) {
            return;
        }
        
        this.chatHistoryList.innerHTML = '';
        
        if (this.chatHistories.length === 0) {
            return;
        }
        
        this.chatHistories.forEach((history, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.dataset.historyId = history.id;
            
            historyItem.innerHTML = `
                <div class="history-item-content">
                    <span class="history-item-title">${this.escapeHtml(history.title)}</span>
                </div>
                <button class="history-item-delete" data-history-id="${history.id}" title="删除">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            `;
            
            // 点击历史项加载对话
            historyItem.addEventListener('click', (e) => {
                if (!e.target.closest('.history-item-delete')) {
                    this.loadChatHistory(history.id);
                }
            });
            
            // 删除历史对话
            const deleteBtn = historyItem.querySelector('.history-item-delete');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteChatHistory(history.id);
            });
            
            this.chatHistoryList.appendChild(historyItem);
        });
    }
    
    // 加载历史对话
    async loadChatHistory(historyId) {
        const history = this.chatHistories.find(h => h.id === historyId);
        if (!history) {
            return;
        }

        this.switchMainView('chat');
        
        // 如果当前有对话内容，且不是同一个对话，先保存
        if (this.currentChatHistory.length > 0 && this.sessionId !== historyId) {
            if (this.isCurrentChatFromHistory) {
                // 如果当前对话也是从历史记录加载的，更新它
                this.updateCurrentChatHistory();
            } else {
                // 如果当前对话是新对话，保存为新历史
                this.saveCurrentChat();
            }
        }
        
        try {
            // 从后端获取会话历史
            const response = await fetch(`/api/chat/session/${historyId}`);
            if (response.ok) {
                const data = await response.json();
                const backendHistory = data.history || [];
                
                // 更新会话ID
                this.sessionId = history.id;
                this.isCurrentChatFromHistory = true;
                
                // 清空并重新渲染消息
                if (this.chatMessages) {
                    this.chatMessages.innerHTML = '';
                    
                    // 如果后端有历史记录，使用后端的
                    if (backendHistory.length > 0) {
                        this.currentChatHistory = [];
                        backendHistory.forEach(msg => {
                            // 后端返回格式: {role: "user|assistant", content: "...", timestamp: "..."}
                            const messageType = msg.role === 'user' ? 'user' : 'bot';
                            this.addMessage(messageType, msg.content, false, false);
                        });
                    } else {
                        // 否则使用localStorage的历史记录
                        this.currentChatHistory = [...history.messages];
                        history.messages.forEach(msg => {
                            this.addMessage(msg.type, msg.content, false, false);
                        });
                    }
                }
            } else {
                // 如果后端请求失败，使用localStorage的历史记录
                console.warn('从后端加载历史失败，使用本地缓存');
                this.sessionId = history.id;
                this.currentChatHistory = [...history.messages];
                this.isCurrentChatFromHistory = true;
                
                if (this.chatMessages) {
                    this.chatMessages.innerHTML = '';
                    history.messages.forEach(msg => {
                        this.addMessage(msg.type, msg.content, false, false);
                    });
                }
            }
        } catch (error) {
            console.error('加载会话历史失败:', error);
            // 出错时使用localStorage的历史记录
            this.sessionId = history.id;
            this.currentChatHistory = [...history.messages];
            this.isCurrentChatFromHistory = true;
            
            if (this.chatMessages) {
                this.chatMessages.innerHTML = '';
                history.messages.forEach(msg => {
                    this.addMessage(msg.type, msg.content, false, false);
                });
            }
        }
        
        // 更新UI
        this.checkAndSetCentered();
        this.renderChatHistory();
    }
    
    // 删除历史对话
    async deleteChatHistory(historyId) {
        try {
            // 调用后端API清空会话
            const response = await fetch('/api/chat/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: historyId
                })
            });

            if (!response.ok) {
                throw new Error('清空会话失败');
            }

            const result = await response.json();
            
            if (result.status === 'success') {
                // 从本地存储中删除
                this.chatHistories = this.chatHistories.filter(h => h.id !== historyId);
                this.saveChatHistories();
                this.renderChatHistory();
                
                // 如果删除的是当前对话，清空当前对话
                if (this.sessionId === historyId) {
                    this.currentChatHistory = [];
                    if (this.chatMessages) {
                        this.chatMessages.innerHTML = '';
                    }
                    this.sessionId = this.generateSessionId();
                    this.checkAndSetCentered();
                }
                
                this.showNotification('会话已清空', 'success');
            } else {
                throw new Error(result.message || '清空会话失败');
            }
        } catch (error) {
            console.error('删除历史对话失败:', error);
            this.showNotification('删除失败: ' + error.message, 'error');
        }
    }

    // 切换模式下拉菜单
    toggleModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) {
                wrapper.classList.toggle('active');
            }
        }
    }

    // 关闭模式下拉菜单
    closeModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) {
                wrapper.classList.remove('active');
            }
        }
    }

    // 选择模式
    selectMode(mode) {
        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成后再切换模式', 'warning');
            return;
        }
        
        this.currentMode = mode;
        this.updateUI();
        
        const modeNames = {
            'quick': '快速',
            'stream': '流式'
        };
        
        this.showNotification(`已切换到${modeNames[mode]}模式`, 'info');
    }

    // 更新UI
    updateUI() {
        // 更新模式选择器显示
        if (this.currentModeText) {
            const modeNames = {
                'quick': '快速',
                'stream': '流式'
            };
            this.currentModeText.textContent = modeNames[this.currentMode] || '快速';
        }
        
        // 更新下拉菜单选中状态
        const dropdownItems = document.querySelectorAll('.dropdown-item');
        dropdownItems.forEach(item => {
            const mode = item.getAttribute('data-mode');
            if (mode === this.currentMode) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        // 更新发送按钮状态
        if (this.sendButton) {
            this.sendButton.disabled = this.isStreaming;
        }
        
        // 更新输入框状态
        if (this.messageInput) {
            this.messageInput.disabled = this.isStreaming;
            this.messageInput.placeholder = '问问智能OnCall助手';
        }

        if (this.governanceRefreshBtn) {
            this.governanceRefreshBtn.disabled = this.isGovernanceLoading || this.isChunkPublishing;
        }

        this.updateMainViewState();
        this.updateChunkEditorActions();
    }

    updateMainViewState() {
        const isGovernance = this.currentMainView === 'governance';

        if (this.chatContainer) {
            this.chatContainer.classList.toggle('hidden', isGovernance);
        }

        if (this.governanceView) {
            this.governanceView.classList.toggle('hidden', !isGovernance);
        }

        if (this.aiOpsSidebarBtn) {
            this.aiOpsSidebarBtn.classList.toggle('hidden', isGovernance);
        }

        if (this.governanceEntryBtn) {
            this.governanceEntryBtn.classList.toggle('active', isGovernance);
        }
    }

    switchMainView(viewName) {
        this.currentMainView = viewName === 'governance' ? 'governance' : 'chat';
        this.updateMainViewState();

        if (this.currentMainView === 'governance') {
            this.renderGovernanceView();
            if (this.fingerprintList.length === 0 && !this.isGovernanceLoading) {
                this.loadFingerprintList({ withOverlay: true });
            }
        }
    }

    renderGovernanceView() {
        this.renderFingerprintList(this.fingerprintList);
        this.renderFingerprintDetail(this.currentFingerprintDetail);
        this.renderChunkEditor(this.currentChunk);
    }

    async governanceRequest(path, options = {}) {
        const response = await fetch(`${this.apiBaseUrl}${path}`, options);
        let payload = null;

        try {
            payload = await response.json();
        } catch (error) {
            payload = null;
        }

        if (!response.ok) {
            const message = payload?.detail || payload?.message || `HTTP错误: ${response.status}`;
            throw new Error(message);
        }

        if (payload?.code !== 200 && payload?.message !== 'success') {
            throw new Error(payload?.message || '请求失败');
        }

        return payload?.data;
    }

    setGovernanceLoading(isLoading, message = '正在加载低置信度治理数据...', withOverlay = false) {
        this.isGovernanceLoading = isLoading;

        if (withOverlay) {
            if (isLoading) {
                this.showLoadingOverlay(true, {
                    title: message,
                    subtitle: '请稍候',
                });
            } else {
                this.showLoadingOverlay(false);
            }
        }

        this.updateUI();
    }

    setChunkPublishing(isPublishing) {
        this.isChunkPublishing = isPublishing;
        if (isPublishing) {
            this.showLoadingOverlay(true, {
                title: '正在发布修订内容...',
                subtitle: '系统正在同步向量库与知识库，请稍候',
            });
        } else {
            this.showLoadingOverlay(false);
        }
        this.updateUI();
    }

    encodeChunkKey(chunkKey) {
        return encodeURIComponent(chunkKey || '');
    }

    encodeFingerprint(fingerprint) {
        return encodeURIComponent(fingerprint || '');
    }

    async loadFingerprintList({ force = false, withOverlay = false } = {}) {
        if (this.isGovernanceLoading) {
            return;
        }

        if (!force && this.fingerprintList.length > 0) {
            this.renderFingerprintList(this.fingerprintList);
            return;
        }

        this.setGovernanceLoading(true, '正在加载低置信度问题...', withOverlay);

        if (this.governanceFingerprintList) {
            this.governanceFingerprintList.innerHTML = '<div class="governance-loading-block">正在加载低置信度问题...</div>';
        }

        try {
            const data = await this.governanceRequest('/admin/low-confidence/fingerprints');
            this.fingerprintList = Array.isArray(data?.items) ? data.items : [];
            this.renderFingerprintList(this.fingerprintList);

            if (!this.currentFingerprint && this.fingerprintList.length > 0) {
                await this.openFingerprint(this.fingerprintList[0].query_fingerprint);
            } else if (this.currentFingerprint) {
                const stillExists = this.fingerprintList.some(item => item.query_fingerprint === this.currentFingerprint);
                if (!stillExists) {
                    this.currentFingerprint = null;
                    this.currentFingerprintDetail = null;
                    this.currentChunk = null;
                    this.currentChunkHistory = [];
                    this.renderFingerprintDetail(null);
                    this.renderChunkEditor(null);
                }
            }
        } catch (error) {
            console.error('加载 fingerprint 列表失败:', error);
            this.renderFingerprintListError(error.message);
            this.showNotification(`加载低置信度问题失败: ${error.message}`, 'error');
        } finally {
            this.setGovernanceLoading(false, '', withOverlay);
        }
    }

    async loadFingerprintDetail(fingerprint) {
        if (!fingerprint) {
            return;
        }

        if (this.governanceDetail) {
            this.governanceDetail.innerHTML = '<div class="governance-loading-block">正在加载指纹详情...</div>';
        }

        try {
            const data = await this.governanceRequest(`/admin/low-confidence/fingerprints/${this.encodeFingerprint(fingerprint)}`);
            this.currentFingerprint = fingerprint;
            this.currentFingerprintDetail = data;
            this.renderFingerprintList(this.fingerprintList);
            this.renderFingerprintDetail(data);
        } catch (error) {
            console.error('加载 fingerprint 详情失败:', error);
            this.currentFingerprintDetail = null;
            this.renderFingerprintDetailError(error.message);
            this.showNotification(`加载指纹详情失败: ${error.message}`, 'error');
        }
    }

    async loadChunkDetail(chunkKey) {
        if (!chunkKey) {
            return;
        }

        this.currentChunk = null;
        this.currentChunkHistory = [];
        this.isDraftDirty = false;
        this.renderChunkEditorLoading();

        try {
            const data = await this.governanceRequest(`/admin/chunks/${this.encodeChunkKey(chunkKey)}`);
            this.currentChunk = data;
            this.isDraftDirty = false;
            this.renderChunkEditor(data);
        } catch (error) {
            console.error('加载 chunk 详情失败:', error);
            this.renderChunkEditorError(error.message);
            this.showNotification(`加载 chunk 详情失败: ${error.message}`, 'error');
        }
    }

    async loadChunkHistory(chunkKey, { force = false } = {}) {
        if (!chunkKey) {
            return;
        }

        if (!force && this.currentChunkHistory.length > 0 && this.currentChunk?.chunk_key === chunkKey) {
            this.renderChunkHistory(this.currentChunkHistory);
            return;
        }

        this.renderChunkHistoryLoading();

        try {
            const data = await this.governanceRequest(`/admin/chunks/${this.encodeChunkKey(chunkKey)}/history`);
            this.currentChunkHistory = Array.isArray(data?.items) ? data.items : [];
            this.renderChunkHistory(this.currentChunkHistory);
        } catch (error) {
            console.error('加载 chunk 历史失败:', error);
            this.renderChunkHistoryError(error.message);
            this.showNotification(`加载历史失败: ${error.message}`, 'error');
        }
    }

    async openFingerprint(fingerprint) {
        if (!fingerprint) {
            return;
        }

        this.currentFingerprint = fingerprint;
        this.currentChunk = null;
        this.currentChunkHistory = [];
        this.renderFingerprintList(this.fingerprintList);
        this.renderChunkEditor(null);
        await this.loadFingerprintDetail(fingerprint);
    }

    async openChunkEditor(chunkKey) {
        if (!chunkKey) {
            return;
        }

        await this.loadChunkDetail(chunkKey);
        await this.loadChunkHistory(chunkKey);
    }

    async saveChunkDraft(chunkKey, { silent = false } = {}) {
        if (!chunkKey) {
            return false;
        }

        const draftTextarea = document.getElementById('chunkDraftTextarea');
        const draftText = draftTextarea ? draftTextarea.value : this.currentChunk?.draft_text || '';

        try {
            const data = await this.governanceRequest(`/admin/chunks/${this.encodeChunkKey(chunkKey)}/draft`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    draftText,
                }),
            });

            this.currentChunk = data;
            this.isDraftDirty = false;
            this.renderChunkEditor(data);
            if (!silent) {
                this.showNotification('草稿已保存', 'success');
            }
            return true;
        } catch (error) {
            console.error('保存草稿失败:', error);
            this.showNotification(`保存草稿失败: ${error.message}`, 'error');
            return false;
        }
    }

    async publishChunk(chunkKey) {
        if (!chunkKey || this.isChunkPublishing) {
            return;
        }

        if (this.isDraftDirty) {
            const saved = await this.saveChunkDraft(chunkKey, { silent: true });
            if (!saved) {
                return;
            }
        }

        this.setChunkPublishing(true);

        try {
            const data = await this.governanceRequest(`/admin/chunks/${this.encodeChunkKey(chunkKey)}/publish`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    editor: 'frontend-admin',
                    editNote: '治理工作台手动发布',
                }),
            });

            this.currentChunk = data;
            this.isDraftDirty = false;
            this.renderChunkEditor(data);
            await this.loadChunkHistory(chunkKey, { force: true });
            this.showNotification('Chunk 发布成功', 'success');
        } catch (error) {
            console.error('发布 chunk 失败:', error);
            this.showNotification(`发布失败: ${error.message}`, 'error');
            if (this.currentChunk?.chunk_key === chunkKey) {
                await this.loadChunkDetail(chunkKey);
                await this.loadChunkHistory(chunkKey, { force: true });
            }
        } finally {
            this.setChunkPublishing(false);
        }
    }

    renderFingerprintList(items = []) {
        if (!this.governanceFingerprintList) {
            return;
        }

        if (!items.length) {
            this.governanceFingerprintList.innerHTML = '<div class="governance-empty">暂无低置信度问题</div>';
            return;
        }

        this.governanceFingerprintList.innerHTML = items.map((item) => {
            const fingerprint = item.query_fingerprint || '';
            const isActive = fingerprint === this.currentFingerprint;

            return `
                <button class="fingerprint-item${isActive ? ' active' : ''}" data-fingerprint="${this.escapeHtml(fingerprint)}" type="button">
                    <div class="fingerprint-item-head">
                        <span class="fingerprint-count">${this.escapeHtml(String(item.event_count ?? 0))}</span>
                        <span class="fingerprint-updated">${this.escapeHtml(this.formatDateTime(item.last_seen_at))}</span>
                    </div>
                    <div class="fingerprint-title">${this.escapeHtml(item.normalized_query || '未命名问题')}</div>
                    <div class="fingerprint-meta">${this.escapeHtml(fingerprint)}</div>
                    <div class="fingerprint-range">首次: ${this.escapeHtml(this.formatDateTime(item.first_seen_at))}</div>
                </button>
            `;
        }).join('');
    }

    renderFingerprintListError(message) {
        if (!this.governanceFingerprintList) {
            return;
        }

        this.governanceFingerprintList.innerHTML = `
            <div class="governance-error">
                <div>低置信度问题加载失败</div>
                <div class="governance-error-detail">${this.escapeHtml(message || '未知错误')}</div>
            </div>
        `;
    }

    renderFingerprintDetail(data) {
        if (!this.governanceDetail) {
            return;
        }

        if (!data) {
            if (this.governanceDetailHint) {
                this.governanceDetailHint.textContent = '选择左侧问题查看详情';
            }
            this.governanceDetail.innerHTML = '<div class="governance-empty">暂未选择问题指纹</div>';
            return;
        }

        if (this.governanceDetailHint) {
            this.governanceDetailHint.textContent = `${data.eventCount || 0} 个低置信度事件`;
        }

        const events = Array.isArray(data.events) ? data.events : [];
        if (!events.length) {
            this.governanceDetail.innerHTML = '<div class="governance-empty">该指纹下暂无事件</div>';
            return;
        }

        this.governanceDetail.innerHTML = `
            <div class="fingerprint-summary-card">
                <div class="fingerprint-summary-label">当前指纹</div>
                <div class="fingerprint-summary-title">${this.escapeHtml(events[0]?.normalized_query || data.fingerprint || '')}</div>
                <div class="fingerprint-summary-meta">${this.escapeHtml(data.fingerprint || '')}</div>
            </div>
            ${events.map((eventItem, index) => {
                const chunks = Array.isArray(eventItem.chunks) ? eventItem.chunks : [];
                return `
                    <section class="event-card">
                        <div class="event-card-head">
                            <div>
                                <div class="event-badge">事件 ${index + 1}</div>
                                <h3>${this.escapeHtml(eventItem.raw_query || '未记录原始问题')}</h3>
                            </div>
                            <div class="event-confidence-pill ${this.escapeHtml(eventItem.overall_confidence || 'unknown')}">${this.escapeHtml(eventItem.overall_confidence || 'unknown')}</div>
                        </div>
                        <div class="event-meta-grid">
                            <div><span>时间</span><strong>${this.escapeHtml(this.formatDateTime(eventItem.created_at))}</strong></div>
                            <div><span>归一化问题</span><strong>${this.escapeHtml(eventItem.normalized_query || '-')}</strong></div>
                            <div><span>原因</span><strong>${this.escapeHtml(eventItem.low_confidence_reason || eventItem.reason || '-')}</strong></div>
                        </div>
                        <div class="event-chunk-list">
                            ${chunks.length ? chunks.map((chunk) => `
                                <button class="chunk-snapshot-card" data-chunk-key="${this.escapeHtml(chunk.chunk_key_snapshot || '')}" type="button">
                                    <div class="chunk-snapshot-head">
                                        <span class="chunk-file-name">${this.escapeHtml(chunk.file_name_snapshot || '未知文件')}</span>
                                        <span class="chunk-score">${this.escapeHtml(this.formatScore(chunk.rerank_score))}</span>
                                    </div>
                                    <div class="chunk-snapshot-meta">
                                        <span>页码 ${this.escapeHtml(this.formatNullable(chunk.page_number_snapshot))}</span>
                                        <span>${this.escapeHtml(chunk.section_path_snapshot || '未标注章节')}</span>
                                        <span class="chunk-confidence-tag ${this.escapeHtml(chunk.document_confidence || 'unknown')}">${this.escapeHtml(chunk.document_confidence || 'unknown')}</span>
                                    </div>
                                    <div class="chunk-snapshot-text markdown-surface">${this.renderMarkdown(chunk.chunk_text_snapshot || '无快照内容')}</div>
                                </button>
                            `).join('') : '<div class="governance-empty compact">暂无 chunk 快照</div>'}
                        </div>
                    </section>
                `;
            }).join('')}
        `;

        this.governanceDetail.querySelectorAll('.markdown-surface').forEach((node) => {
            this.highlightCodeBlocks(node);
        });
    }

    renderFingerprintDetailError(message) {
        if (!this.governanceDetail) {
            return;
        }

        if (this.governanceDetailHint) {
            this.governanceDetailHint.textContent = '详情加载失败';
        }

        this.governanceDetail.innerHTML = `
            <div class="governance-error">
                <div>指纹详情加载失败</div>
                <div class="governance-error-detail">${this.escapeHtml(message || '未知错误')}</div>
            </div>
        `;
    }

    renderChunkEditorLoading() {
        if (this.chunkEditorHint) {
            this.chunkEditorHint.textContent = '正在加载 chunk 内容...';
        }
        if (this.chunkEditorContent) {
            this.chunkEditorContent.innerHTML = '<div class="governance-loading-block">正在加载 chunk 内容...</div>';
        }
    }

    renderChunkEditor(chunk) {
        if (!this.chunkEditorContent) {
            return;
        }

        if (!chunk) {
            if (this.chunkEditorHint) {
                this.chunkEditorHint.textContent = '选择某个 chunk 开始修订';
            }
            this.chunkEditorContent.innerHTML = '<div class="governance-empty">暂未选择 chunk</div>';
            return;
        }

        const metadata = chunk.metadata || {};
        const publishedText = chunk.published_text || chunk.source_text || '';
        const draftText = chunk.draft_text ?? publishedText;

        if (this.chunkEditorHint) {
            this.chunkEditorHint.textContent = `${chunk.file_name || metadata._file_name || '未知文件'} · v${chunk.published_version || 0}`;
        }

        this.chunkEditorContent.innerHTML = `
            <div class="chunk-editor-meta-card">
                <div class="chunk-editor-meta-grid">
                    <div><span>文件</span><strong>${this.escapeHtml(chunk.file_name || metadata._file_name || '-')}</strong></div>
                    <div><span>集合</span><strong>${this.escapeHtml(chunk.collection_name || '-')}</strong></div>
                    <div><span>页码</span><strong>${this.escapeHtml(this.formatNullable(chunk.page_number ?? metadata.page_number))}</strong></div>
                    <div><span>状态</span><strong class="sync-status-tag ${this.escapeHtml(chunk.sync_status || 'unknown')}">${this.escapeHtml(chunk.sync_status || 'unknown')}</strong></div>
                    <div><span>版本</span><strong>${this.escapeHtml(String(chunk.published_version || 0))}</strong></div>
                    <div><span>章节</span><strong>${this.escapeHtml(chunk.section_path || metadata.section_path || '-')}</strong></div>
                </div>
                <div class="chunk-key-line">${this.escapeHtml(chunk.chunk_key || '')}</div>
                ${chunk.last_publish_error ? `<div class="publish-error-banner">${this.escapeHtml(chunk.last_publish_error)}</div>` : ''}
            </div>

            <div class="chunk-text-section">
                <div class="chunk-text-header">
                    <h3>当前线上内容</h3>
                </div>
                <div class="chunk-text-preview markdown-surface">${this.renderMarkdown(publishedText || '暂无已发布内容')}</div>
            </div>

            <div class="chunk-text-section">
                <div class="chunk-text-header">
                    <h3>草稿内容</h3>
                    <span class="draft-state ${this.isDraftDirty ? 'dirty' : 'saved'}">${this.isDraftDirty ? '未保存修改' : '已同步到页面'}</span>
                </div>
                <textarea id="chunkDraftTextarea" class="chunk-draft-textarea" ${this.isChunkPublishing ? 'disabled' : ''} placeholder="请输入修订后的 chunk 文本">${this.escapeHtml(draftText)}</textarea>
            </div>

            <div class="chunk-action-row">
                <button class="governance-primary-btn" type="button" data-action="save-draft">保存草稿</button>
                <button class="governance-accent-btn" type="button" data-action="publish-chunk">发布</button>
                <button class="governance-secondary-btn" type="button" data-action="view-history">刷新历史</button>
            </div>

            <section class="chunk-history-section">
                <div class="chunk-text-header">
                    <h3>发布历史</h3>
                </div>
                <div id="chunkHistoryPanel">
                    <div class="governance-empty compact">暂无历史记录</div>
                </div>
            </section>
        `;

        this.chunkEditorContent.querySelectorAll('.markdown-surface').forEach((node) => {
            this.highlightCodeBlocks(node);
        });
        this.renderChunkHistory(this.currentChunkHistory);
        this.updateChunkEditorActions();
    }

    renderChunkEditorError(message) {
        if (!this.chunkEditorContent) {
            return;
        }

        if (this.chunkEditorHint) {
            this.chunkEditorHint.textContent = 'Chunk 加载失败';
        }

        this.chunkEditorContent.innerHTML = `
            <div class="governance-error">
                <div>Chunk 加载失败</div>
                <div class="governance-error-detail">${this.escapeHtml(message || '未知错误')}</div>
            </div>
        `;
    }

    renderChunkHistoryLoading() {
        const historyPanel = document.getElementById('chunkHistoryPanel');
        if (historyPanel) {
            historyPanel.innerHTML = '<div class="governance-loading-block compact">正在加载历史...</div>';
        }
    }

    renderChunkHistory(items = []) {
        const historyPanel = document.getElementById('chunkHistoryPanel');
        if (!historyPanel) {
            return;
        }

        if (!items.length) {
            historyPanel.innerHTML = '<div class="governance-empty compact">暂无发布历史</div>';
            return;
        }

        historyPanel.innerHTML = `
            <div class="chunk-history-list">
                ${items.map((item) => `
                    <article class="history-entry">
                        <div class="history-entry-head">
                            <strong>v${this.escapeHtml(String(item.version_no || '-'))}</strong>
                            <span>${this.escapeHtml(this.formatDateTime(item.created_at))}</span>
                        </div>
                        <div class="history-entry-meta">
                            <span>编辑人: ${this.escapeHtml(item.editor || 'admin')}</span>
                            <span>状态: ${this.escapeHtml(item.publish_status || 'success')}</span>
                        </div>
                        ${item.edit_note ? `<div class="history-entry-note">${this.escapeHtml(item.edit_note)}</div>` : ''}
                        ${item.new_text ? `<div class="history-entry-preview markdown-surface">${this.renderMarkdown(item.new_text)}</div>` : ''}
                    </article>
                `).join('')}
            </div>
        `;

        historyPanel.querySelectorAll('.markdown-surface').forEach((node) => {
            this.highlightCodeBlocks(node);
        });
    }

    renderChunkHistoryError(message) {
        const historyPanel = document.getElementById('chunkHistoryPanel');
        if (!historyPanel) {
            return;
        }

        historyPanel.innerHTML = `
            <div class="governance-error compact">
                <div>历史加载失败</div>
                <div class="governance-error-detail">${this.escapeHtml(message || '未知错误')}</div>
            </div>
        `;
    }

    updateChunkEditorActions() {
        if (!this.chunkEditorContent) {
            return;
        }

        const saveBtn = this.chunkEditorContent.querySelector('[data-action="save-draft"]');
        const publishBtn = this.chunkEditorContent.querySelector('[data-action="publish-chunk"]');
        const historyBtn = this.chunkEditorContent.querySelector('[data-action="view-history"]');
        const draftTextarea = this.chunkEditorContent.querySelector('#chunkDraftTextarea');

        const hasChunk = Boolean(this.currentChunk);
        const disabled = this.isChunkPublishing || !hasChunk;

        if (saveBtn) {
            saveBtn.disabled = disabled;
            saveBtn.textContent = this.isChunkPublishing ? '处理中...' : (this.isDraftDirty ? '保存草稿' : '草稿已保存');
        }

        if (publishBtn) {
            publishBtn.disabled = disabled;
            publishBtn.textContent = this.isChunkPublishing ? '发布中...' : '发布';
        }

        if (historyBtn) {
            historyBtn.disabled = disabled;
        }

        if (draftTextarea) {
            draftTextarea.disabled = disabled;
        }
    }

    formatDateTime(value) {
        if (!value) {
            return '-';
        }

        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
            return String(value);
        }

        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
        });
    }

    formatScore(value) {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric.toFixed(3) : '-';
    }

    formatNullable(value) {
        if (value === null || value === undefined || value === '') {
            return '-';
        }
        return String(value);
    }

    // 生成随机会话ID
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    // 发送消息
    async sendMessage() {
        let message = '';
        if (this.messageInput) {
            message = this.messageInput.value.trim();
        }
        
        if (!message) {
            this.showNotification('请输入消息内容', 'warning');
            return;
        }

        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成', 'warning');
            return;
        }

        // 显示用户消息
        this.addMessage('user', message);
        
        // 清空输入框
        if (this.messageInput) {
            this.messageInput.value = '';
        }

        // 设置发送状态
        this.isStreaming = true;
        this.updateUI();

        try {
            if (this.currentMode === 'quick') {
                await this.sendQuickMessage(message);
            } else if (this.currentMode === 'stream') {
                await this.sendStreamMessage(message);
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            this.addMessage('assistant', '抱歉，发送消息时出现错误：' + error.message);
        } finally {
            this.isStreaming = false;
            this.updateUI();
            
            // 如果当前对话是从历史记录加载的，更新历史记录
            if (this.isCurrentChatFromHistory && this.currentChatHistory.length > 0) {
                this.updateCurrentChatHistory();
                this.renderChatHistory(); // 更新历史对话列表显示
            }
        }
    }

    // 发送快速消息（普通对话）
    async sendQuickMessage(message) {
        // 添加等待提示消息
        const loadingMessage = this.addLoadingMessage('正在思考...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    Id: this.sessionId,
                    Question: message
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            console.log('[sendQuickMessage] 响应数据:', JSON.stringify(data));
            
            // 移除等待提示消息
            if (loadingMessage && loadingMessage.parentNode) {
                loadingMessage.parentNode.removeChild(loadingMessage);
            }
            
            // 统一响应格式：检查 data.code 或 data.message 判断请求是否成功
            if (data.code === 200 || data.message === 'success') {
                // data.data 是 ChatResponse 对象
                const chatResponse = data.data;
                
                if (chatResponse && chatResponse.success) {
                    // 成功：添加实际响应消息（即使 answer 为空也显示）
                    const answer = chatResponse.answer || '（无回复内容）';
                    this.addMessage('assistant', answer);
                } else if (chatResponse && chatResponse.errorMessage) {
                    // 业务错误
                    throw new Error(chatResponse.errorMessage);
                } else {
                    // 兜底：尝试显示任何可用内容
                    const fallbackAnswer = chatResponse?.answer || chatResponse?.errorMessage || '服务返回了空内容';
                    this.addMessage('assistant', fallbackAnswer);
                }
            } else {
                // HTTP 成功但业务失败
                throw new Error(data.message || '请求失败');
            }
        } catch (error) {
            // 出错时也要移除等待提示消息
            if (loadingMessage && loadingMessage.parentNode) {
                loadingMessage.parentNode.removeChild(loadingMessage);
            }
            throw error;
        }
    }

    // 发送流式消息
    async sendStreamMessage(message) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/chat_stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    Id: this.sessionId,
                    Question: message
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }
            
            // 创建助手消息元素
            const assistantMessageElement = this.addMessage('assistant', '', true);
            let fullResponse = '';

            // 处理流式响应
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = '';

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) {
                        // 流结束，使用统一的处理方法
                        this.handleStreamComplete(assistantMessageElement, fullResponse);
                        break;
                    }

                    // 解码数据并添加到缓冲区
                    buffer += decoder.decode(value, { stream: true });
                    
                    // 按行分割处理
                    const lines = buffer.split('\n');
                    // 保留最后一行（可能不完整）
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.trim() === '') continue;
                        
                        console.log('[SSE调试] 收到行:', line);
                        
                        // 解析SSE格式
                        if (line.startsWith('id:')) {
                            console.log('[SSE调试] 解析到ID');
                            continue;
                        } else if (line.startsWith('event:')) {
                            // 兼容 "event:message" 和 "event: message" 两种格式
                            currentEvent = line.substring(6).trim();
                            console.log('[SSE调试] 解析到事件类型:', currentEvent);
                            // 注意：后端统一使用 "message" 事件名，真正的类型在 data 的 JSON 中
                            continue;
                        } else if (line.startsWith('data:')) {
                            // 兼容 "data:xxx" 和 "data: xxx" 两种格式
                            const rawData = line.substring(5).trim();
                            console.log('[SSE调试] 解析到数据, currentEvent:', currentEvent, ', rawData:', rawData);
                            
                            // 兼容旧格式 [DONE] 标记
                            if (rawData === '[DONE]') {
                                // 流结束标记，将内容转换为Markdown渲染
                                this.handleStreamComplete(assistantMessageElement, fullResponse);
                                return;
                            }
                            
                            // 处理 SSE 数据
                            try {
                                // 尝试解析为 SseMessage 格式的 JSON
                                const sseMessage = JSON.parse(rawData);
                                console.log('[SSE调试] 解析JSON成功:', sseMessage);
                                
                                if (sseMessage && typeof sseMessage.type === 'string') {
                                    if (sseMessage.type === 'content') {
                                        const content = sseMessage.data || '';
                                        fullResponse += content;
                                        console.log('[SSE调试] 添加内容:', content);
                                        
                                        // 实时渲染 Markdown
                                        if (assistantMessageElement) {
                                            const messageContent = assistantMessageElement.querySelector('.message-content');
                                            messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                            // 高亮代码块
                                            this.highlightCodeBlocks(messageContent);
                                            this.scrollToBottom();
                                        }
                                    } else if (sseMessage.type === 'done') {
                                        console.log('[SSE调试] 收到done标记，流结束');
                                        this.handleStreamComplete(assistantMessageElement, fullResponse);
                                        return;
                                    } else if (sseMessage.type === 'error') {
                                        console.error('[SSE调试] 收到错误:', sseMessage.data);
                                        if (assistantMessageElement) {
                                            const messageContent = assistantMessageElement.querySelector('.message-content');
                                            messageContent.innerHTML = this.renderMarkdown('错误: ' + (sseMessage.data || '未知错误'));
                                        }
                                        return;
                                    }
                                } else {
                                    // 不是标准 SseMessage 格式，尝试兼容处理
                                    console.log('[SSE调试] 非标准格式，尝试兼容处理');
                                    fullResponse += rawData;
                                    if (assistantMessageElement) {
                                        const messageContent = assistantMessageElement.querySelector('.message-content');
                                        messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                        this.highlightCodeBlocks(messageContent);
                                        this.scrollToBottom();
                                    }
                                }
                            } catch (e) {
                                // JSON 解析失败，尝试兼容旧格式
                                console.log('[SSE调试] JSON解析失败，使用兼容模式:', e.message);
                                if (rawData === '') {
                                    fullResponse += '\n';
                                } else {
                                    fullResponse += rawData;
                                }
                                
                                if (assistantMessageElement) {
                                    const messageContent = assistantMessageElement.querySelector('.message-content');
                                    messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                    this.highlightCodeBlocks(messageContent);
                                    this.scrollToBottom();
                                }
                            }
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        } catch (error) {
            throw error;
        }
    }

    // 添加消息到聊天界面
    addMessage(type, content, isStreaming = false, saveToHistory = true) {
        // 检查是否是第一条消息，如果是则移除居中样式
        const isFirstMessage = this.chatMessages && this.chatMessages.querySelectorAll('.message').length === 0;
        
        // 保存消息到当前对话历史（如果不是流式消息且需要保存）
        if (!isStreaming && saveToHistory && content) {
            this.currentChatHistory.push({
                type: type,
                content: content,
                timestamp: new Date().toISOString()
            });
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}${isStreaming ? ' streaming' : ''}`;

        // 如果是assistant消息，添加头像图标
        if (type === 'assistant') {
            const messageAvatar = document.createElement('div');
            messageAvatar.className = 'message-avatar';
            messageAvatar.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/>
                </svg>
            `;
            messageDiv.appendChild(messageAvatar);
        }

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // 如果是assistant消息且不是流式消息，使用Markdown渲染
        if (type === 'assistant' && !isStreaming) {
            messageContent.innerHTML = this.renderMarkdown(content);
            // 高亮代码块
            this.highlightCodeBlocks(messageContent);
        } else {
            // 用户消息或流式消息使用纯文本
            messageContent.textContent = content;
        }

        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            
            // 如果是第一条消息，移除居中样式并添加动画
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                // 添加动画类
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            
            this.scrollToBottom();
        }

        return messageDiv;
    }

    // 添加带加载动画的消息
    addLoadingMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        // 添加头像图标
        const messageAvatar = document.createElement('div');
        messageAvatar.className = 'message-avatar';
        messageAvatar.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/>
            </svg>
        `;
        messageDiv.appendChild(messageAvatar);

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content loading-message-content';
        
        // 创建文本和动画容器
        const textSpan = document.createElement('span');
        textSpan.textContent = content;
        
        // 创建旋转动画图标
        const loadingIcon = document.createElement('span');
        loadingIcon.className = 'loading-spinner-icon';
        loadingIcon.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" fill="currentColor" opacity="0.2"/>
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10c1.54 0 3-.36 4.28-1l-1.5-2.6C13.64 19.62 12.84 20 12 20c-4.41 0-8-3.59-8-8s3.59-8 8-8c.84 0 1.64.38 2.18 1l1.5-2.6C13 2.36 12.54 2 12 2z" fill="currentColor"/>
            </svg>
        `;
        
        messageContent.appendChild(textSpan);
        messageContent.appendChild(loadingIcon);
        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            
            // 如果是第一条消息，移除居中样式
            const isFirstMessage = this.chatMessages.querySelectorAll('.message').length === 1;
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            
            this.scrollToBottom();
        }

        return messageDiv;
    }
    
    // 检查并设置居中样式
    checkAndSetCentered() {
        if (this.chatMessages && this.chatContainer) {
            const hasMessages = this.chatMessages.querySelectorAll('.message').length > 0;
            if (!hasMessages) {
                this.chatContainer.classList.add('centered');
            } else {
                this.chatContainer.classList.remove('centered');
            }
        }
    }

    // 滚动到底部
    scrollToBottom() {
        if (this.chatMessages) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }
    }

    // 处理流式传输完成
    handleStreamComplete(assistantMessageElement, fullResponse) {
        if (assistantMessageElement) {
            assistantMessageElement.classList.remove('streaming');
            const messageContent = assistantMessageElement.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = this.renderMarkdown(fullResponse);
                // 高亮代码块
                this.highlightCodeBlocks(messageContent);
            }
        }
        // 保存流式消息到历史记录
        if (fullResponse) {
            this.currentChatHistory.push({
                type: 'assistant',
                content: fullResponse,
                timestamp: new Date().toISOString()
            });
            // 如果当前对话是从历史记录加载的，更新历史记录
            if (this.isCurrentChatFromHistory) {
                this.updateCurrentChatHistory();
                this.renderChatHistory();
            }
        }
    }

    // 显示通知
    showNotification(message, type = 'info') {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 10000;
            animation: slideIn 0.3s ease;
            max-width: 300px;
        `;

        // 根据类型设置颜色（Google Material Design配色）
        const colors = {
            info: '#1a73e8',
            success: '#34a853',
            warning: '#fbbc04',
            error: '#ea4335'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        // 添加到页面
        document.body.appendChild(notification);

        // 3秒后自动移除
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    // 处理文件选择
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            // 验证文件格式
            if (!this.validateFileType(file)) {
                this.showNotification('只支持上传 TXT、Markdown (.md) 或 PDF 格式的文件', 'error');
                this.fileInput.value = '';
                return;
            }
            this.uploadFile(file);
        }
    }

    // 验证文件类型
    validateFileType(file) {
        const fileName = file.name.toLowerCase();
        const allowedExtensions = ['.txt', '.md', '.markdown', '.pdf'];
        return allowedExtensions.some(ext => fileName.endsWith(ext));
    }

    // 上传文件到知识库
    async uploadFile(file) {
        // 再次验证文件类型（双重保险）
        if (!this.validateFileType(file)) {
            this.showNotification('只支持上传 TXT、Markdown (.md) 或 PDF 格式的文件', 'error');
            return;
        }

        // 验证文件大小（限制为50MB）
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('文件大小不能超过50MB', 'error');
            return;
        }

        // 锁定前端并显示上传遮罩层
        this.isStreaming = true;
        this.updateUI();
        this.showUploadOverlay(true, file.name);

        try {
            // 创建 FormData
            const formData = new FormData();
            formData.append('file', file);

            // 发送上传请求
            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();

            if ((data.code === 200 || data.message === 'success') && data.data) {
                // 在聊天界面显示上传成功消息
                const successMessage = `${file.name} 上传到知识库成功`;
                this.addMessage('assistant', successMessage, false, true);
            } else {
                throw new Error(data.message || '上传失败');
            }
        } catch (error) {
            console.error('文件上传失败:', error);
            this.showNotification('文件上传失败: ' + error.message, 'error');
        } finally {
            // 清空文件输入
            if (this.fileInput) {
                this.fileInput.value = '';
            }
            // 解锁前端
            this.isStreaming = false;
            this.showUploadOverlay(false);
            this.updateUI();
        }
    }

    // 格式化文件大小
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    // 发送智能运维请求（SSE 流式模式）
    async sendAIOpsRequest(loadingMessageElement) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/aiops`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            let fullResponse = '';

            // 处理 SSE 流式响应
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = 'message'; // 默认事件类型为 message

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    
                    if (done) {
                        // 流结束，更新最终内容
                        if (fullResponse) {
                            console.log('AI Ops 流结束，更新最终内容，长度:', fullResponse.length);
                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                        }
                        break;
                    }

                    // 解码数据并添加到缓冲区
                    buffer += decoder.decode(value, { stream: true });
                    
                    // 按行分割处理
                    const lines = buffer.split('\n');
                    // 保留最后一行（可能不完整）
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.trim() === '') continue;
                        
                        console.log('[AI Ops SSE] 收到行:', line);
                        
                        // 解析 SSE 格式
                        if (line.startsWith('id:')) {
                            continue;
                        } else if (line.startsWith('event:')) {
                            currentEvent = line.substring(6).trim();
                            console.log('[AI Ops SSE] 事件类型:', currentEvent);
                            continue;
                        } else if (line.startsWith('data:')) {
                            const rawData = line.substring(5).trim();
                            console.log('[AI Ops SSE] 数据:', rawData, ', currentEvent:', currentEvent);
                            
                            // 解析可能包含多个JSON对象的数据
                            const processJsonMessages = (data) => {
                                const jsonPattern = /\{"type"\s*:\s*"[^"]+"\s*,\s*"data"\s*:\s*(?:"[^"]*"|null)\}/g;
                                const matches = data.match(jsonPattern);
                                
                                if (matches && matches.length > 0) {
                                    console.log('[AI Ops SSE] 匹配到', matches.length, '个JSON对象');
                                    for (const jsonStr of matches) {
                                        try {
                                            const sseMessage = JSON.parse(jsonStr);
                                            if (sseMessage.type === 'content') {
                                                fullResponse += sseMessage.data || '';
                                            } else if (sseMessage.type === 'plan') {
                                                // 处理计划创建事件
                                                const planText = `\n\n## 📋 执行计划\n${sseMessage.message}\n\n`;
                                                fullResponse += planText;
                                            } else if (sseMessage.type === 'step_complete') {
                                                // 处理步骤完成事件
                                                const stepText = `\n✅ ${sseMessage.message}\n`;
                                                fullResponse += stepText;
                                            } else if (sseMessage.type === 'status') {
                                                // 处理状态更新事件
                                                const statusText = `\n⏳ ${sseMessage.message}\n`;
                                                fullResponse += statusText;
                                            } else if (sseMessage.type === 'report') {
                                                // 处理最终报告事件 - 流式输出
                                                console.log('AI Ops 最终报告生成');
                                                const reportText = `\n\n## 🎯 诊断报告\n\n${sseMessage.report || ''}\n`;
                                                fullResponse += reportText;
                                            } else if (sseMessage.type === 'complete') {
                                                // 处理完成事件
                                                console.log('AI Ops 诊断完成');
                                                if (sseMessage.response) {
                                                    fullResponse += `\n\n${sseMessage.response}`;
                                                }
                                                this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                                return true;
                                            } else if (sseMessage.type === 'done') {
                                                console.log('AI Ops 流完成，最终内容长度:', fullResponse.length);
                                                this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                                return true;
                                            } else if (sseMessage.type === 'error') {
                                                throw new Error(sseMessage.data || sseMessage.message || '智能运维分析失败');
                                            }
                                        } catch (e) {
                                            if (e.message.includes('智能运维')) throw e;
                                            console.log('[AI Ops SSE] 单个JSON解析失败:', jsonStr);
                                        }
                                    }
                                    if (loadingMessageElement) {
                                        this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                    }
                                    return false;
                                }
                                return null;
                            };
                            
                            const result = processJsonMessages(rawData);
                            if (result === true) {
                                return; // 流结束
                            } else if (result === null) {
                                // 没有匹配到多个JSON，尝试单个JSON解析
                                try {
                                    const sseMessage = JSON.parse(rawData);
                                    if (sseMessage && sseMessage.type) {
                                        if (sseMessage.type === 'content') {
                                            fullResponse += sseMessage.data || '';
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'plan') {
                                            // 处理计划创建事件
                                            const planText = `\n\n## 📋 执行计划\n${sseMessage.message}\n\n`;
                                            fullResponse += planText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'step_complete') {
                                            // 处理步骤完成事件
                                            const stepText = `\n✅ ${sseMessage.message}\n`;
                                            fullResponse += stepText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'status') {
                                            // 处理状态更新事件
                                            const statusText = `\n⏳ ${sseMessage.message}\n`;
                                            fullResponse += statusText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'report') {
                                            // 处理最终报告事件 - 这是关键！
                                            console.log('AI Ops 最终报告生成，流式输出中...');
                                            const reportText = `\n\n## 🎯 诊断报告\n\n${sseMessage.report || ''}\n`;
                                            fullResponse += reportText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'complete') {
                                            // 处理完成事件
                                            console.log('AI Ops 诊断完成，最终内容长度:', fullResponse.length);
                                            if (sseMessage.response) {
                                                fullResponse += `\n\n${sseMessage.response}`;
                                            }
                                            // 使用最终的完整内容更新消息
                                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                            return;
                                        } else if (sseMessage.type === 'done') {
                                            console.log('AI Ops 流完成，最终内容长度:', fullResponse.length);
                                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                            return;
                                        } else if (sseMessage.type === 'error') {
                                            throw new Error(sseMessage.data || sseMessage.message || '智能运维分析失败');
                                        }
                                    } else {
                                        fullResponse += rawData;
                                        if (loadingMessageElement) {
                                            this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                        }
                                    }
                                } catch (e) {
                                    if (e.message.includes('智能运维')) throw e;
                                    // 非 JSON 格式，直接追加原始数据
                                    fullResponse += rawData;
                                    if (loadingMessageElement) {
                                        this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                    }
                                }
                            }
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        } catch (error) {
            throw error;
        }
    }

    // 更新智能运维流式内容（实时显示）
    updateAIOpsStreamContent(messageElement, content) {
        if (!messageElement) return;
        
        // 添加 aiops-message 类
        messageElement.classList.add('aiops-message');
        
        const messageContentWrapper = messageElement.querySelector('.message-content-wrapper');
        if (messageContentWrapper) {
            let messageContent = messageContentWrapper.querySelector('.message-content');
            if (!messageContent) {
                messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContentWrapper.appendChild(messageContent);
            }
            // 流式显示时使用纯文本
            messageContent.textContent = content;
            this.scrollToBottom();
        }
    }

    // 更新智能运维消息（带折叠详情）
    updateAIOpsMessage(messageElement, response, details) {
        console.log('updateAIOpsMessage 被调用');
        console.log('messageElement:', messageElement);
        console.log('response:', response);
        console.log('response length:', response ? response.length : 0);
        console.log('details:', details);
        
        if (!messageElement) {
            // 如果没有传入消息元素，则创建新消息
            console.log('messageElement 为空，创建新消息');
            return this.addAIOpsMessage(response, details);
        }

        // 添加aiops-message类
        messageElement.classList.add('aiops-message');

        // 获取消息内容包装器
        const messageContentWrapper = messageElement.querySelector('.message-content-wrapper');
        if (!messageContentWrapper) {
            console.error('未找到 message-content-wrapper');
            return;
        }

        // 清空现有内容（保留消息内容容器）
        const messageContent = messageContentWrapper.querySelector('.message-content');
        if (!messageContent) {
            console.error('未找到 message-content');
            return;
        }

        // 移除加载动画相关的类和内容
        messageContent.classList.remove('loading-message-content');
        messageContent.textContent = '';
        
        // 移除加载图标（如果存在）
        const loadingIcon = messageContent.querySelector('.loading-spinner-icon');
        if (loadingIcon) {
            loadingIcon.remove();
        }

        // 详情部分（可折叠）- 先显示
        if (details && details.length > 0) {
            // 检查是否已存在详情容器
            let detailsContainer = messageElement.querySelector('.aiops-details');
            if (!detailsContainer) {
                detailsContainer = document.createElement('div');
                detailsContainer.className = 'aiops-details';
                messageContentWrapper.insertBefore(detailsContainer, messageContent);
            } else {
                // 清空现有详情
                detailsContainer.innerHTML = '';
            }

            const detailsToggle = document.createElement('div');
            detailsToggle.className = 'details-toggle';
            detailsToggle.innerHTML = `
                <svg class="toggle-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 18L15 12L9 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>查看详细步骤 (${details.length}条)</span>
            `;

            const detailsContent = document.createElement('div');
            detailsContent.className = 'details-content';
            
            details.forEach((detail, index) => {
                const detailItem = document.createElement('div');
                detailItem.className = 'detail-item';
                detailItem.innerHTML = `<strong>步骤 ${index + 1}:</strong> ${this.escapeHtml(detail)}`;
                detailsContent.appendChild(detailItem);
            });

            // 点击切换折叠状态
            detailsToggle.addEventListener('click', () => {
                detailsContent.classList.toggle('expanded');
                detailsToggle.classList.toggle('expanded');
            });

            detailsContainer.appendChild(detailsToggle);
            detailsContainer.appendChild(detailsContent);
        }

        // 更新主要响应内容（使用Markdown渲染）
        console.log('开始渲染 Markdown');
        const renderedHtml = this.renderMarkdown(response);
        console.log('Markdown 渲染完成，HTML 长度:', renderedHtml ? renderedHtml.length : 0);
        messageContent.innerHTML = renderedHtml;
        console.log('innerHTML 已设置');
        // 高亮代码块
        this.highlightCodeBlocks(messageContent);
        console.log('代码块高亮完成');
        
        // 保存到历史记录
        this.currentChatHistory.push({
            type: 'assistant',
            content: response,
            timestamp: new Date().toISOString()
        });
        
        this.scrollToBottom();
        return messageElement;
    }

    // 添加智能运维消息（带折叠详情）- 保留用于兼容性
    addAIOpsMessage(response, details) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant aiops-message';

        // 添加头像图标
        const messageAvatar = document.createElement('div');
        messageAvatar.className = 'message-avatar';
        messageAvatar.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/>
            </svg>
        `;
        messageDiv.appendChild(messageAvatar);

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        // 详情部分（可折叠）- 先显示
        if (details && details.length > 0) {
            const detailsContainer = document.createElement('div');
            detailsContainer.className = 'aiops-details';

            const detailsToggle = document.createElement('div');
            detailsToggle.className = 'details-toggle';
            detailsToggle.innerHTML = `
                <svg class="toggle-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 18L15 12L9 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>查看详细步骤 (${details.length}条)</span>
            `;

            const detailsContent = document.createElement('div');
            detailsContent.className = 'details-content';
            
            details.forEach((detail, index) => {
                const detailItem = document.createElement('div');
                detailItem.className = 'detail-item';
                detailItem.innerHTML = `<strong>步骤 ${index + 1}:</strong> ${this.escapeHtml(detail)}`;
                detailsContent.appendChild(detailItem);
            });

            // 点击切换折叠状态
            detailsToggle.addEventListener('click', () => {
                detailsContent.classList.toggle('expanded');
                detailsToggle.classList.toggle('expanded');
            });

            detailsContainer.appendChild(detailsToggle);
            detailsContainer.appendChild(detailsContent);
            messageContentWrapper.appendChild(detailsContainer);
        }

        // 主要响应内容 - 后显示（使用Markdown渲染）
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = this.renderMarkdown(response);
        // 高亮代码块
        this.highlightCodeBlocks(messageContent);
        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);
        
        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            this.scrollToBottom();
        }

        return messageDiv;
    }

    // HTML转义
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // 触发智能运维（点击智能运维按钮时直接调用）
    async triggerAIOps() {
        if (this.isStreaming) {
            this.showNotification('请等待当前操作完成', 'warning');
            return;
        }

        this.switchMainView('chat');

        // 新建对话
        this.newChat();
        
        // 添加"分析中..."的消息（带旋转动画）
        const loadingMessage = this.addLoadingMessage('分析中...');
        this.currentAIOpsMessage = loadingMessage; // 保存消息引用用于后续更新
        
        // 设置发送状态
        this.isStreaming = true;
        this.updateUI();

        try {
            await this.sendAIOpsRequest(loadingMessage);
        } catch (error) {
            console.error('智能运维分析失败:', error);
            // 更新消息为错误信息
            if (loadingMessage) {
                const messageContent = loadingMessage.querySelector('.message-content');
                if (messageContent) {
                    messageContent.textContent = '抱歉，智能运维分析时出现错误：' + error.message;
                }
            }
        } finally {
            this.isStreaming = false;
            this.currentAIOpsMessage = null;
            this.updateUI();
        }
    }

    // 显示/隐藏加载遮罩层
    showLoadingOverlay(show, options = {}) {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.style.display = 'flex';
                const loadingText = this.loadingOverlay.querySelector('.loading-text');
                const loadingSubtext = this.loadingOverlay.querySelector('.loading-subtext');
                if (loadingText) loadingText.textContent = options.title || '智能运维分析中，请稍候...';
                if (loadingSubtext) loadingSubtext.textContent = options.subtitle || '后端正在处理，请耐心等待';
                // 防止页面滚动
                document.body.style.overflow = 'hidden';
            } else {
                this.loadingOverlay.style.display = 'none';
                // 恢复页面滚动
                document.body.style.overflow = '';
            }
        }
    }

    // 显示/隐藏上传遮罩层
    showUploadOverlay(show, fileName = '') {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.style.display = 'flex';
                // 更新文字为上传中
                const loadingText = this.loadingOverlay.querySelector('.loading-text');
                const loadingSubtext = this.loadingOverlay.querySelector('.loading-subtext');
                if (loadingText) loadingText.textContent = '正在上传文件...';
                if (loadingSubtext) loadingSubtext.textContent = fileName ? `上传: ${fileName}` : '请稍候';
                // 防止页面滚动
                document.body.style.overflow = 'hidden';
            } else {
                this.loadingOverlay.style.display = 'none';
                // 恢复页面滚动
                document.body.style.overflow = '';
            }
        }
    }
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new SuperBizAgentApp();
});
