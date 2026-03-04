// Vetinari Web UI - Main Application
// Enhanced UX Version with Responsive & Micro-interactions

// ==================== UTILITY FUNCTIONS ====================

/**
 * Debounce: delay execution until after `wait` ms since last call.
 */
function debounce(fn, wait) {
    let timer;
    return function(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), wait);
    };
}

/**
 * loadCurrentProject: refresh the currently open project if one is active.
 */
function loadCurrentProject() {
    if (typeof currentProjectId !== 'undefined' && currentProjectId) {
        loadProjectDetails(currentProjectId);
    }
}

// ==================== RESPONSIVE SIDEBAR MANAGEMENT ====================

class SidebarManager {
    constructor() {
        this.sidebar = document.querySelector('.sidebar');
        this.mainContent = document.querySelector('.main-content');
        this.toggleBtn = document.getElementById('sidebarToggle');
        this.isCollapsed = this.loadCollapsedState();
        this.isMobile = window.innerWidth < 768;
        
        this.init();
    }
    
    init() {
        this.applyCollapsedState();
        this.attachEventListeners();
        this.handleResponsive();
    }
    
    loadCollapsedState() {
        return localStorage.getItem('sidebarCollapsed') === 'true';
    }
    
    saveCollapsedState(state) {
        localStorage.setItem('sidebarCollapsed', state);
    }
    
    applyCollapsedState() {
        if (this.isCollapsed && !this.isMobile) {
            this.sidebar.classList.add('collapsed');
            this.mainContent.classList.add('sidebar-collapsed');
        } else if (this.isMobile) {
            this.sidebar.classList.remove('open');
        }
    }
    
    toggle() {
        if (this.isMobile) {
            this.sidebar.classList.toggle('open');
            const isOpen = this.sidebar.classList.contains('open');
            this.toggleBtn.setAttribute('aria-expanded', isOpen);
        } else {
            this.isCollapsed = !this.isCollapsed;
            this.sidebar.classList.toggle('collapsed');
            this.mainContent.classList.toggle('sidebar-collapsed');
            this.saveCollapsedState(this.isCollapsed);
            this.toggleBtn.setAttribute('aria-expanded', !this.isCollapsed);
        }
    }
    
    attachEventListeners() {
        this.toggleBtn.addEventListener('click', () => this.toggle());
        document.addEventListener('click', (e) => {
            if (this.isMobile && !e.target.closest('.sidebar') && !e.target.closest('#sidebarToggle')) {
                this.sidebar.classList.remove('open');
            }
        });
    }
    
    handleResponsive() {
        window.addEventListener('resize', () => {
            const wasMobile = this.isMobile;
            this.isMobile = window.innerWidth < 768;
            
            if (wasMobile && !this.isMobile) {
                this.sidebar.classList.remove('open');
                if (this.isCollapsed) {
                    this.sidebar.classList.add('collapsed');
                    this.mainContent.classList.add('sidebar-collapsed');
                }
            } else if (!wasMobile && this.isMobile) {
                this.sidebar.classList.remove('collapsed');
                this.mainContent.classList.remove('sidebar-collapsed');
                this.sidebar.classList.remove('open');
            }
        });
    }
}

// Initialize sidebar manager
let sidebarManager;
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        sidebarManager = new SidebarManager();
    });
} else {
    sidebarManager = new SidebarManager();
}

// ==================== ACCESSIBILITY ENHANCEMENTS ====================

class AccessibilityManager {
    constructor() {
        this.prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        this.init();
    }
    
    init() {
        this.enhanceKeyboardNavigation();
        this.setupReducedMotionToggle();
    }
    
    enhanceKeyboardNavigation() {
        // Add keyboard navigation for sidebar
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + B to toggle sidebar
            if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
                e.preventDefault();
                sidebarManager?.toggle();
            }
            // Escape to close mobile sidebar
            if (e.key === 'Escape' && sidebarManager?.isMobile) {
                document.querySelector('.sidebar')?.classList.remove('open');
            }
        });
        
        // Ensure all interactive elements are keyboard accessible
        const interactiveElements = document.querySelectorAll('.nav-item, .btn, input, textarea, select');
        interactiveElements.forEach(el => {
            if (!el.hasAttribute('tabindex') && el.tagName !== 'BUTTON' && el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA' && el.tagName !== 'SELECT') {
                el.setAttribute('tabindex', '0');
            }
        });
    }
    
    setupReducedMotionToggle() {
        if (this.prefersReducedMotion) {
            document.documentElement.style.setProperty('--transition-fast', '0ms');
            document.documentElement.style.setProperty('--transition-base', '0ms');
            document.documentElement.style.setProperty('--transition-slow', '0ms');
            document.documentElement.style.setProperty('--transition-slowest', '0ms');
        }
    }
}

// Initialize accessibility manager
let a11yManager;
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        a11yManager = new AccessibilityManager();
    });
} else {
    a11yManager = new AccessibilityManager();
}

// ==================== PREFERENCES MANAGER ====================

class PreferencesManager {
    constructor() {
        this.reducedMotionToggle = document.getElementById('reducedMotionToggle');
        this.compactModeToggle = document.getElementById('compactModeToggle');
        
        if (this.reducedMotionToggle || this.compactModeToggle) {
            this.loadPreferences();
            this.attachEventListeners();
        }
    }
    
    loadPreferences() {
        const reducedMotion = localStorage.getItem('reducedMotion') === 'true';
        const compactMode = localStorage.getItem('compactMode') === 'true';
        
        if (this.reducedMotionToggle) {
            this.reducedMotionToggle.checked = reducedMotion;
        }
        if (this.compactModeToggle) {
            this.compactModeToggle.checked = compactMode;
        }
        
        this.applyPreferences(reducedMotion, compactMode);
    }
    
    applyPreferences(reducedMotion, compactMode) {
        const root = document.documentElement;
        
        if (reducedMotion) {
            root.style.setProperty('--transition-fast', '0ms');
            root.style.setProperty('--transition-base', '0ms');
            root.style.setProperty('--transition-slow', '0ms');
            root.style.setProperty('--transition-slowest', '0ms');
        } else {
            root.style.removeProperty('--transition-fast');
            root.style.removeProperty('--transition-base');
            root.style.removeProperty('--transition-slow');
            root.style.removeProperty('--transition-slowest');
        }
        
        if (compactMode) {
            document.body.classList.add('compact-mode');
        } else {
            document.body.classList.remove('compact-mode');
        }
    }
    
    attachEventListeners() {
        if (this.reducedMotionToggle) {
            this.reducedMotionToggle.addEventListener('change', (e) => {
                localStorage.setItem('reducedMotion', e.target.checked);
                this.applyPreferences(e.target.checked, document.getElementById('compactModeToggle')?.checked || false);
            });
        }
        
        if (this.compactModeToggle) {
            this.compactModeToggle.addEventListener('change', (e) => {
                localStorage.setItem('compactMode', e.target.checked);
                this.applyPreferences(document.getElementById('reducedMotionToggle')?.checked || false, e.target.checked);
            });
        }
    }
}

// Initialize preferences manager
let prefsManager;
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        prefsManager = new PreferencesManager();
    });
} else {
    prefsManager = new PreferencesManager();
}

// Safe JSON parsing helper
function safeJsonParse(response) {
    return response.json().catch(err => {
        console.error('JSON parse error:', err);
        return { error: 'Invalid JSON response', details: err.message };
    });
}

// Theme management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const newTheme = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    if (btn) {
        const icon = btn.querySelector('i');
        if (icon) {
            icon.className = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
        }
    }
}

// State
let currentView = 'dashboard';
let models = [];
let tasks = [];
let activityLog = [];
let sidebarCollapsed = false;

// Status Banner Functions
function showStatusBanner(message, type = 'info') {
    const banner = document.getElementById('statusBanner');
    const text = document.getElementById('statusBannerText');
    const content = document.getElementById('statusBannerContent');
    
    if (banner && text && content) {
        banner.style.display = 'flex';
        banner.className = 'status-banner ' + type;
        text.textContent = message;
        
        // Auto-hide after 5 seconds for info
        if (type === 'info') {
            setTimeout(() => {
                hideStatusBanner();
            }, 5000);
        }
    }
}

function hideStatusBanner() {
    const banner = document.getElementById('statusBanner');
    if (banner) {
        banner.style.display = 'none';
    }
}

window.hideStatusBanner = hideStatusBanner;

// Initialize - everything inside DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNavigation();
    initSidebar();
    initAdvancedNav();
    // Load dashboard first — it fetches models and updates _lastModelCount
    // checkLMStudioConnection then reads _lastModelCount without extra fetch
    loadDashboard().then(() => checkLMStudioConnection());
    loadSettings();
    loadSidebarProjects();

    // Auto-refresh — dashboard every 60s (includes model check), projects every 15s
    setInterval(loadDashboard, 60000);
    setInterval(loadSidebarProjects, 15000);
    setInterval(checkLMStudioConnection, 90000);
    
    // Output dropdown change handler
    document.getElementById('outputTaskSelect')?.addEventListener('change', loadOutputForTask);
    
    // Model swap on prompt page
    document.getElementById('chatModelSelect')?.addEventListener('change', async (e) => {
        const modelId = e.target.value;
        if (!modelId || !currentProjectId) return;
        
        try {
            const res = await fetch('/api/swap-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ project_id: currentProjectId, model_id: modelId })
            });
            const data = await safeJsonParse(res);
            if (data.status === 'swapped') {
                addActivity(`Swapped to model: ${modelId}`);
            }
        } catch (error) {
            console.error('Error swapping model:', error);
        }
    });
    
    // Archive filters
    document.getElementById('archiveSearch')?.addEventListener('input', loadArchive);
    document.getElementById('archiveStatusFilter')?.addEventListener('change', loadArchive);
    
    // Theme toggle
    document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
    
    // Header buttons
    document.getElementById('discoverBtn')?.addEventListener('click', async () => {
        const btn = document.getElementById('discoverBtn');
        if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>'; }
        addActivity('Discovering models...');
        try {
            // Force-refresh bypasses the TTL cache
            const res = await fetch('/api/models/refresh', { method: 'POST' });
            const data = await safeJsonParse(res);
            if (data.error) {
                addActivity('Discovery error: ' + data.error, 'error');
            } else {
                addActivity(`Found ${data.count || data.discovered || 0} models`);
                _lastModelCount = data.count || 0;
                updateLMStatusBar(true, _lastModelCount, '');
            }
            loadModels();
            loadDashboard();
        } catch (error) {
            addActivity('Error discovering models', 'error');
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-compass"></i>'; }
        }
    });
    
    document.getElementById('runAllBtn')?.addEventListener('click', async () => {
        const userGoal = window.prompt('Enter your goal for the project:');
        if (!userGoal) return;
        
        addActivity(`Creating project with goal: ${userGoal.substring(0, 50)}...`);
        showStatusBanner('Creating project and planning tasks...', 'info');
        
        const btn = document.getElementById('runAllBtn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating...';
        }
        
        try {
            const res = await fetch('/api/new-project', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ goal: userGoal, auto_run: true })
            });
            const data = await safeJsonParse(res);
            
            if (data.error) {
                addActivity('Error creating project: ' + data.error, 'error');
                showStatusBanner('Error: ' + data.error, 'error');
            } else {
                addActivity(`Project created: ${data.project_id} with ${data.tasks?.length || 0} tasks`);
                
                // Show warnings if any
                if (data.warnings && data.warnings.length > 0) {
                    showStatusBanner(data.warnings[0], 'warning');
                } else {
                    showStatusBanner('Project created! Tasks are running.', 'success');
                }
                
                // Show the plan
                if (data.tasks && data.tasks.length > 0) {
                    let planMsg = 'Tasks:\n';
                    data.tasks.forEach(t => {
                        planMsg += `- ${t.id}: ${t.description} (${t.assigned_model_id})\n`;
                    });
                    addActivity(planMsg);
                }
                
                // Refresh projects
                loadSidebarProjects();
                
                // Switch to workflow view to show progress
                switchView('workflow');
            }
            loadDashboard();
        } catch (error) {
            addActivity('Error creating project: ' + error, 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-play"></i> Run All Tasks';
            }
        }
    });
    
    // Model view buttons
    document.getElementById('refreshModels')?.addEventListener('click', loadModels);
    
    // Workflow view buttons
    document.getElementById('refreshWorkflow')?.addEventListener('click', loadWorkflow);
    
    // Chat buttons
    document.getElementById('sendMessageBtn')?.addEventListener('click', sendChatMessage);
    document.getElementById('newProjectBtn')?.addEventListener('click', () => {
        currentProjectId = null;
        document.getElementById('chatMessages').innerHTML = `
            <div class="chat-empty">
                <i class="fas fa-comments"></i>
                <p>Type your goal below to start a new project</p>
            </div>
        `;
        document.getElementById('chatTasks').style.display = 'none';
        document.querySelectorAll('.project-item').forEach(item => {
            item.classList.remove('active');
        });
        document.getElementById('chatInput').focus();
    });
    
    // Chat input enter key
    document.getElementById('chatInput')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            sendChatMessage();
        }
    });
    
    // Quick prompt on dashboard
    document.getElementById('quickPromptBtn')?.addEventListener('click', async () => {
        const input = document.getElementById('quickPromptInput');
        const select = document.getElementById('quickModelSelect');
        const prompt = input?.value.trim();
        const model = select?.value;
        
        if (!prompt) {
            addActivity('Please enter a prompt', 'error');
            return;
        }
        
        // Switch to prompt view and create project
        switchView('prompt');
        
        setTimeout(async () => {
            currentProjectId = null;
            document.getElementById('chatInput').value = prompt;
            if (model) {
                document.getElementById('chatModelSelect').value = model;
            }
            await sendChatMessage();
        }, 200);
    });
    
    // Quick prompt enter key
    document.getElementById('quickPromptInput')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            document.getElementById('quickPromptBtn')?.click();
        }
    });
    
    // System prompt management
    document.getElementById('toggleSystemPromptBtn')?.addEventListener('click', () => {
        const panel = document.getElementById('systemPromptPanel');
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    });
    
    document.getElementById('systemPromptSelect')?.addEventListener('change', (e) => {
        const input = document.getElementById('systemPromptInput');
        if (e.target.value) {
            input.value = e.target.value;
            currentSystemPrompt = e.target.value;
        } else {
            input.value = 'You are a helpful coding assistant.';
            currentSystemPrompt = '';
        }
    });
    
    document.getElementById('systemPromptInput')?.addEventListener('input', (e) => {
        currentSystemPrompt = e.target.value;
    });
    
    document.getElementById('newSystemPromptBtn')?.addEventListener('click', async () => {
        const name = prompt('Enter a name for this system prompt preset:');
        if (!name) return;
        
        const content = document.getElementById('systemPromptInput')?.value || 'You are a helpful coding assistant.';
        
        try {
            const res = await fetch('/api/system-prompts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, content })
            });
            const data = await safeJsonParse(res);
            
            if (data.status === 'saved') {
                addActivity('System prompt saved as: ' + name);
                loadPrompt();
            } else if (data.error) {
                addActivity('Error saving prompt: ' + data.error, 'error');
            }
        } catch (error) {
            addActivity('Error saving prompt', 'error');
        }
    });
    
    document.getElementById('deleteSystemPromptBtn')?.addEventListener('click', async () => {
        const select = document.getElementById('systemPromptSelect');
        const selectedOption = select?.selectedOptions[0];
        
        if (!selectedOption || !selectedOption.value) {
            addActivity('No preset selected to delete', 'error');
            return;
        }
        
        const name = selectedOption.text;
        if (!confirm(`Delete system prompt preset "${name}"?`)) return;
        
        try {
            const res = await fetch(`/api/system-prompts/${encodeURIComponent(name)}`, {
                method: 'DELETE'
            });
            const data = await safeJsonParse(res);
            
            if (data.status === 'deleted') {
                addActivity('System prompt deleted');
                loadPrompt();
            } else if (data.error) {
                addActivity('Error deleting prompt: ' + data.error, 'error');
            }
        } catch (error) {
            addActivity('Error deleting prompt', 'error');
        }
    });
});

// Sidebar
function initSidebar() {
    const toggleBtn = document.getElementById('sidebarToggle');
    toggleBtn?.addEventListener('click', () => {
        sidebarCollapsed = !sidebarCollapsed;
        document.querySelector('.sidebar').classList.toggle('collapsed', sidebarCollapsed);
        document.querySelector('.main-content').classList.toggle('sidebar-collapsed', sidebarCollapsed);
    });
}

// Navigation
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            if (view) switchView(view);
        });
    });
}

// Advanced nav section toggle
function initAdvancedNav() {
    const toggle = document.getElementById('advancedNavToggle');
    const items = document.getElementById('advancedNavItems');
    const chevron = document.getElementById('advancedNavChevron');
    if (!toggle || !items) return;

    toggle.addEventListener('click', () => {
        const expanded = toggle.getAttribute('aria-expanded') === 'true';
        toggle.setAttribute('aria-expanded', !expanded);
        items.style.display = expanded ? 'none' : 'flex';
        items.style.flexDirection = 'column';
        items.style.gap = '4px';
        if (chevron) {
            chevron.style.transform = expanded ? '' : 'rotate(90deg)';
        }
    });

    // Auto-expand if an advanced view is active
    const activeAdvanced = items.querySelector('.nav-item.active');
    if (activeAdvanced) {
        toggle.setAttribute('aria-expanded', 'true');
        items.style.display = 'flex';
    }
}

// LM Studio connection status — uses /api/status only (no extra discovery call)
async function checkLMStudioConnection() {
    try {
        const res = await fetch('/api/status');
        const status = await safeJsonParse(res);
        // Re-use the model count from the dashboard data if available; otherwise
        // rely on whatever was last fetched. Avoids a duplicate /api/models call.
        const modelCount = _lastModelCount !== undefined ? _lastModelCount : 0;
        updateLMStatusBar(status.status === 'running', modelCount, status.host || '');
    } catch (e) {
        updateLMStatusBar(false, 0, '');
    }
}

// Tracks the last known model count so checkLMStudioConnection doesn't need to re-fetch
let _lastModelCount = undefined;

function updateLMStatusBar(connected, modelCount, host) {
    // Update the status bar on the dashboard if it exists
    let bar = document.getElementById('lmStatusBar');
    if (!bar) return;
    if (connected) {
        bar.className = 'lm-status-bar connected';
        bar.innerHTML = `<i class="fas fa-circle"></i> <strong>LM Studio connected</strong> — ${modelCount} model${modelCount !== 1 ? 's' : ''} loaded at ${host}`;
    } else {
        bar.className = 'lm-status-bar disconnected';
        bar.innerHTML = `<i class="fas fa-exclamation-circle"></i> <strong>LM Studio not reachable</strong> — Check that LM Studio is running at ${host || 'the configured host'}. <a href="#" onclick="switchView('settings'); return false;">Configure →</a>`;
    }
}

function switchView(view) {
    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.view === view) {
            item.classList.add('active');
        }
    });

    // Auto-expand Advanced nav if switching to an advanced view
    const advancedViews = ['agents', 'output', 'archive', 'tasks', 'decomposition'];
    if (advancedViews.includes(view)) {
        const toggle = document.getElementById('advancedNavToggle');
        const items = document.getElementById('advancedNavItems');
        if (toggle && items && toggle.getAttribute('aria-expanded') === 'false') {
            toggle.setAttribute('aria-expanded', 'true');
            items.style.display = 'flex';
            items.style.flexDirection = 'column';
            items.style.gap = '4px';
            const chevron = document.getElementById('advancedNavChevron');
            if (chevron) chevron.style.transform = 'rotate(90deg)';
        }
    }
    
    // Update view
    document.querySelectorAll('.view').forEach(v => {
        v.classList.remove('active');
    });
    document.getElementById(view + 'View').classList.add('active');
    
    // Update header title
    const titles = {
        dashboard: 'Dashboard',
        models: 'Models',
        tasks: 'Tasks',
        agents: 'Agents',
        output: 'Output',
        archive: 'Archive',
        settings: 'Settings',
        prompt: 'Prompt',
        workflow: 'Workflow',
        decomposition: 'Decomposition Lab'
    };
    const titleEl = document.querySelector('.header-title h1') || document.querySelector('header h1') || document.querySelector('h1');
    if (titleEl) titleEl.textContent = titles[view] || view;
    
    // Show loading indicator
    const mainContent = document.querySelector('.main-content');
    mainContent.classList.add('loading');
    
    // Load view data
    switch(view) {
        case 'dashboard':
            loadDashboard();
            refreshAgentStatus();
            refreshSharedMemory();
            break;
        case 'models':
            loadModels();
            break;
        case 'tasks':
            loadTasks();
            break;
        case 'agents':
            loadAgentsView();
            break;
        case 'output':
            loadOutput();
            break;
        case 'archive':
            loadArchive();
            break;
        case 'settings':
            loadSettings();
            loadModelConfig();
            loadCredentials();
            break;
        case 'prompt':
            loadPrompt();
            break;
        case 'workflow':
            loadWorkflow();
            break;
        case 'decomposition':
            loadDecomposition();
            break;
    }
    
    // Hide loading indicator after a short delay
    setTimeout(() => {
        mainContent.classList.remove('loading');
    }, 300);
    
    currentView = view;
}

// Prompt View - Chat Interface
let currentProjectId = null;
let systemPrompts = [];

async function loadPrompt() {
    const modelSelect = document.getElementById('chatModelSelect');
    const systemPromptSelect = document.getElementById('systemPromptSelect');
    
    try {
        // Load models, system prompts, and projects in parallel
        const [modelsRes, promptsRes, projectsRes] = await Promise.all([
            fetch('/api/models'),
            fetch('/api/system-prompts'),
            fetch('/api/projects')
        ]);
        
        const modelsData = await safeJsonParse(modelsRes);
        const promptsData = await safeJsonParse(promptsRes);
        const projectsData = await safeJsonParse(projectsRes);
        
        // Check for errors
        if (modelsData.error) {
            console.error('Error loading models:', modelsData.error);
            addActivity('Error loading models', 'error');
        }
        if (promptsData.error) {
            console.error('Error loading prompts:', promptsData.error);
        }
        if (projectsData.error) {
            console.error('Error loading projects:', projectsData.error);
        }
        
        models = modelsData.models || [];
        systemPrompts = promptsData.prompts || [];
        
        // Populate model select — use id as value (what LM Studio expects), name as display
        const modelOptions = models.length > 0
            ? models.map(m => `<option value="${m.id || m.name}">${m.name || m.id}</option>`).join('')
            : '<option value="">No models found — click Discover</option>';
        if (modelSelect) modelSelect.innerHTML = modelOptions;
        
        // Populate system prompts select
        let promptOptions = '<option value="">Default System Prompt</option>';
        promptOptions += systemPrompts.map(p => 
            `<option value="${p.content}">${p.name}</option>`
        ).join('');
        if (systemPromptSelect) systemPromptSelect.innerHTML = promptOptions;
        
        // Load projects list
        loadProjectsList(projectsData.projects || []);
        
        // If there's a current project selected, load its details
        if (currentProjectId) {
            const project = (projectsData.projects || []).find(p => p.id === currentProjectId);
            if (project) {
                // Set the model select
                if (modelSelect && project.model) {
                    modelSelect.value = project.model;
                }
                if (project.active_model_id && modelSelect) {
                    modelSelect.value = project.active_model_id;
                }
                // Show tasks if available
                if (project.tasks && project.tasks.length > 0) {
                    renderTasks(project.tasks);
                }
                // Load full project details including conversation
                loadProjectDetails(currentProjectId);
            }
        }
        
    } catch (error) {
        console.error('Error loading prompt view:', error);
        addActivity('Error loading prompt view', 'error');
    }
}

async function loadProjectDetails(projectId) {
    try {
        const res = await fetch(`/api/project/${projectId}`);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            console.error('Error loading project:', data.error);
            return;
        }
        
        if (data.conversation) {
            renderConversation(data.conversation);
        }
        
        if (data.tasks && data.tasks.length > 0) {
            window.currentProjectTasks = data.tasks;
            renderTasks(data.tasks);
        }
        
    } catch (error) {
        console.error('Error loading project details:', error);
    }
}

function loadProjectsList(projects) {
    const list = document.getElementById('projectsList');
    if (!list) return;
    
    if (projects.length === 0) {
        list.innerHTML = '<p style="padding: 1rem; color: var(--secondary);">No projects yet. Click "New" to start!</p>';
        return;
    }
    
    list.innerHTML = projects.map(p => `
        <div class="project-item ${p.id === currentProjectId ? 'active' : ''}" data-id="${p.id}">
            <div class="project-name">${p.name}</div>
            <div class="project-meta">${p.message_count || 0} messages</div>
        </div>
    `).join('');
    
    // Add click handlers
    list.querySelectorAll('.project-item').forEach(item => {
        item.addEventListener('click', () => {
            const projectId = item.dataset.id;
            selectProject(projectId);
        });
    });
}

// Load projects in sidebar (persistent)
async function loadSidebarProjects() {
    const list = document.getElementById('sidebarProjectsList');
    if (!list) return;
    
    try {
        const res = await fetch('/api/projects');
        const data = await safeJsonParse(res);
        
        if (data.error || !data.projects) {
            list.innerHTML = '<p style="padding: 0.5rem; font-size: 0.75rem; color: var(--text-secondary);">No projects</p>';
            return;
        }
        
        const projects = data.projects;
        
        if (projects.length === 0) {
            list.innerHTML = '<p style="padding: 0.5rem; font-size: 0.75rem; color: var(--text-secondary);">No projects yet</p>';
            return;
        }
        
        list.innerHTML = projects.map(p => {
            const status = p.status || 'pending';
            const taskCount = p.tasks?.length || 0;
            const completedCount = p.tasks?.filter(t => t.status === 'completed').length || 0;
            
            return `
                <div class="sidebar-project-item ${p.id === currentProjectId ? 'active' : ''}" data-id="${p.id}" onclick="selectProjectFromSidebar('${p.id}')">
                    <div class="sidebar-project-info">
                        <span class="sidebar-project-name">${p.name}</span>
                        <div class="sidebar-project-status">
                            <span class="status-dot status-${status}"></span>
                            <span>${completedCount}/${taskCount} tasks</span>
                        </div>
                    </div>
                    <div class="project-item-actions">
                        <button class="btn btn-icon btn-secondary" onclick="event.stopPropagation(); quickRename('${p.id}')" title="Rename">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-icon btn-secondary" onclick="event.stopPropagation(); archiveProject('${p.id}')" title="Archive">
                            <i class="fas fa-archive"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        console.error('Error loading sidebar projects:', error);
    }
}

async function quickRename(projectId) {
    const newName = prompt('Enter new project name:');
    if (!newName) return;
    
    try {
        const res = await fetch(`/api/project/${projectId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName })
        });
        const data = await safeJsonParse(res);
        
        if (data.status === 'renamed') {
            addActivity(`Project renamed to: ${newName}`);
            loadSidebarProjects();
        }
    } catch (error) {
        console.error('Error renaming project:', error);
    }
}

window.selectProjectFromSidebar = function(projectId) {
    selectProject(projectId);
    switchView('prompt');
};

async function selectProject(projectId) {
    currentProjectId = projectId;
    
    // Update sidebar selection
    document.querySelectorAll('.project-item').forEach(item => {
        item.classList.toggle('active', item.dataset.id === projectId);
    });
    
    try {
        const res = await fetch(`/api/project/${projectId}`);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            console.error('Error loading project:', data.error);
            return;
        }
        
        if (data.conversation) {
            renderConversation(data.conversation);
        }
        
        if (data.tasks && data.tasks.length > 0) {
            renderTasks(data.tasks);
        }
        
        // Set selected model
        if (data.config && data.config.model) {
            document.getElementById('chatModelSelect').value = data.config.model;
        }
        
    } catch (error) {
        console.error('Error loading project:', error);
    }
}

function renderConversation(conversation, append = false) {
    const messagesEl = document.getElementById('chatMessages');
    if (!messagesEl) return;
    
    if (!conversation || conversation.length === 0) {
        if (!append) {
            messagesEl.innerHTML = `
                <div class="chat-empty">
                    <i class="fas fa-comments"></i>
                    <p>Start a conversation</p>
                </div>
            `;
        }
        return;
    }
    
    // If appending, only add new messages that aren't already displayed
    if (append) {
        const existingMessages = messagesEl.querySelectorAll('.chat-message');
        const existingCount = existingMessages.length;
        const newMessages = conversation.slice(existingCount);
        
        newMessages.forEach(msg => {
            const msgEl = document.createElement('div');
            msgEl.className = `chat-message ${msg.role}`;
            msgEl.innerHTML = `
                <div class="chat-message-avatar">
                    <i class="fas fa-${msg.role === 'user' ? 'user' : 'robot'}"></i>
                </div>
                <div class="chat-message-content">
                    ${formatMessageContent(msg.content)}
                </div>
            `;
            messagesEl.appendChild(msgEl);
        });
    } else {
        // Full render - replace all
        messagesEl.innerHTML = conversation.map(msg => `
            <div class="chat-message ${msg.role}">
                <div class="chat-message-avatar">
                    <i class="fas fa-${msg.role === 'user' ? 'user' : 'robot'}"></i>
                </div>
                <div class="chat-message-content">
                    ${formatMessageContent(msg.content)}
                </div>
            </div>
        `).join('');
    }
    
    // Scroll to bottom
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function formatMessageContent(content) {
    // Convert code blocks
    let formatted = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
    });
    
    // Convert inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Convert newlines to br
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

let currentDepthLimit = 12;
let expandedTasks = new Set();

function renderTasks(tasks) {
    const tasksEl = document.getElementById('chatTasks');
    const tasksList = document.getElementById('tasksList');
    if (!tasksEl || !tasksList) return;
    
    tasksEl.style.display = 'block';
    
    if (!tasks || tasks.length === 0) {
        tasksList.innerHTML = '<p class="no-tasks">No tasks yet</p>';
        return;
    }
    
    const depthLimit = currentDepthLimit;
    
    tasksList.innerHTML = tasks.map(task => {
        const depth = task.depth || 0;
        const isHidden = depth > depthLimit;
        const hasChildren = task.children && task.children.length > 0;
        const isExpanded = expandedTasks.has(task.id);
        const showChildren = isExpanded && hasChildren;
        
        return `
            <div class="task-item ${task.status || 'pending'} depth-${Math.min(depth, 12)} ${isHidden ? 'hidden-depth' : ''}" data-task-id="${task.id}" data-depth="${depth}" onclick="showModelRanking('${task.id}', '${encodeURIComponent(task.description || '')}')">
                ${hasChildren ? `<span class="task-expand" onclick="event.stopPropagation(); toggleTaskExpand('${task.id}')"><i class="fas ${isExpanded ? 'fa-chevron-down' : 'fa-chevron-right'}"></i></span>` : '<span style="width: 1rem; display: inline-block;"></span>'}
                <span class="task-status ${task.output || task.status === 'completed' ? 'completed' : 'pending'}"></span>
                <div class="task-info">
                    <span class="task-name">${task.id}</span>
                    ${task.description ? `<span class="task-desc">${task.description.substring(0, 60)}...</span>` : ''}
                    ${task.assigned_model ? `<span class="task-model"><i class="fas fa-microchip"></i> ${task.assigned_model}</span>` : ''}
                    ${task.model_override ? `<span class="task-override"><i class="fas fa-user"></i> override</span>` : ''}
                </div>
                ${task.files && task.files.length > 0 ? `<span class="task-files">(${task.files.length} files)</span>` : ''}
            </div>
        `;
    }).join('');
}

function toggleTaskExpand(taskId) {
    if (expandedTasks.has(taskId)) {
        expandedTasks.delete(taskId);
    } else {
        expandedTasks.add(taskId);
    }
    loadCurrentProject();
}

function updateDepthView(depth) {
    currentDepthLimit = parseInt(depth);
    document.getElementById('depthValue').textContent = depth;
    renderTasks(window.currentProjectTasks || []);
}

function expandAllTasks() {
    const tasks = window.currentProjectTasks || [];
    tasks.forEach(task => {
        if (task.children && task.children.length > 0) {
            expandedTasks.add(task.id);
        }
    });
    renderTasks(tasks);
}

function collapseAllTasks() {
    expandedTasks.clear();
    renderTasks(window.currentProjectTasks || []);
}

let currentSelectedTaskId = null;

function showModelRanking(taskId, taskDescription) {
    const panel = document.getElementById('modelRankingPanel');
    if (!panel) return;
    
    currentSelectedTaskId = taskId;
    panel.style.display = 'block';
    
    const listEl = document.getElementById('modelRankingList');
    listEl.innerHTML = '<div class="loading"><div class="spinner"></div> Loading models...</div>';
    
    const overrideSection = document.getElementById('modelOverrideSection');
    overrideSection.style.display = 'block';
    
    const description = decodeURIComponent(taskDescription || '');
    
    fetch(`/api/project/${currentProjectId}/model-search`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({task_description: description})
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            listEl.innerHTML = '<p style="color: var(--error);">Error: ' + data.error + '</p>';
            return;
        }
        
        const candidates = data.candidates || [];
        
        listEl.innerHTML = candidates.map((c, i) => `
            <div class="model-candidate-item" onclick="selectOverrideModel('${c.id}', '${c.name}')" onmouseenter="showRationale(this)" onmouseleave="hideRationale(this)">
                <span class="model-candidate-rank">#${i + 1}</span>
                <div class="model-candidate-info">
                    <div class="model-candidate-name">${c.name || c.id}</div>
                    <div class="model-candidate-source">${c.source_type}</div>
                </div>
                <div class="model-candidate-scores">
                    <span class="model-candidate-score hard-data" title="Hard Data">HD: ${(c.hard_data_score || 0).toFixed(2)}</span>
                    <span class="model-candidate-score benchmark" title="Benchmarks">BM: ${(c.benchmark_score || 0).toFixed(2)}</span>
                    <span class="model-candidate-score sentiment" title="Sentiment">SN: ${(c.sentiment_score || 0).toFixed(2)}</span>
                </div>
                <span class="model-candidate-final">${(c.final_score || 0).toFixed(2)}</span>
                <div class="model-candidate-rationale">
                    <strong>Why:</strong> ${c.short_rationale || 'No rationale available'}
                    ${c.provenance && c.provenance.length > 0 ? `<br><a href="${c.provenance[0].url}" target="_blank" style="color: var(--primary-light);">View Source</a>` : ''}
                </div>
            </div>
        `).join('');
        
        const select = document.getElementById('overrideModelSelect');
        select.innerHTML = '<option value="">Select model...</option>' + 
            candidates.map(c => `<option value="${c.id}">${c.name || c.id} (${c.final_score?.toFixed(2)})</option>`).join('');
    })
    .catch(err => {
        listEl.innerHTML = '<p style="color: var(--error);">Error loading models</p>';
    });
}

function showRationale(element) {
    element.classList.add('show-rationale');
}

function hideRationale(element) {
    element.classList.remove('show-rationale');
}

function refreshModelSearch() {
    if (currentSelectedTaskId && window.currentProjectTasks) {
        const task = window.currentProjectTasks.find(t => t.id === currentSelectedTaskId);
        if (task) {
            showModelRanking(currentSelectedTaskId, task.description);
        }
    }
}

function selectOverrideModel(modelId, modelName) {
    const select = document.getElementById('overrideModelSelect');
    select.value = modelId;
}

function applyModelOverride() {
    const select = document.getElementById('overrideModelSelect');
    const modelId = select.value;
    
    if (!modelId || !currentSelectedTaskId) {
        alert('Please select a task and model');
        return;
    }
    
    fetch(`/api/project/${currentProjectId}/task/${currentSelectedTaskId}/override`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model_id: modelId})
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        showStatus('Model override applied successfully');
        
        loadProjectDetails(currentProjectId);
    })
    .catch(err => {
        alert('Error applying override');
    });
}

// Progress rendering functions
function showProgress(tasks) {
    const progressSection = document.getElementById('progressSection');
    if (!progressSection) return;
    
    progressSection.style.display = 'block';
    
    // Calculate overall progress
    const total = tasks.length;
    const completed = tasks.filter(t => t.status === 'completed').length;
    const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    // Update overall progress bar
    const progressBar = document.getElementById('overallProgressBar');
    const progressPercent = document.getElementById('progressPercent');
    if (progressBar) progressBar.style.width = percent + '%';
    if (progressPercent) progressPercent.textContent = percent + '%';
    
    // Render per-task progress
    renderTaskProgress(tasks);
}

function renderTaskProgress(tasks) {
    const tasksEl = document.getElementById('chatTasks');
    if (!tasksEl) return;
    
    // Find or create task progress container
    let progressContainer = document.getElementById('taskProgressContainer');
    if (!progressContainer) {
        const container = document.createElement('div');
        container.id = 'taskProgressContainer';
        container.style.marginTop = '1rem';
        tasksEl.appendChild(container);
        progressContainer = container;
    }
    
    progressContainer.innerHTML = tasks.map(task => {
        const statusClass = task.status === 'completed' ? 'completed' : (task.status === 'running' ? 'running' : 'pending');
        const progress = task.status === 'completed' ? 100 : (task.status === 'running' ? 50 : 0);
        
        let subtasksHtml = '';
        if (task.subtasks && task.subtasks.length > 0) {
            subtasksHtml = `<div class="subtasks" id="subtasks-${task.id}">` + 
                task.subtasks.map(st => `
                    <div class="subtask-item">
                        <span class="task-status ${st.status === 'completed' ? 'completed' : (st.status === 'running' ? 'running' : 'pending')}"></span>
                        <span class="task-id">${st.id}</span>
                        <span>${st.description ? st.description.substring(0, 30) + '...' : ''}</span>
                        <span class="task-model">${st.assigned_model || ''}</span>
                    </div>
                `).join('') + 
                '</div>';
        }
        
        return `
            <div class="task-progress-item">
                <div class="task-progress-header" onclick="toggleSubtasks('${task.id}')">
                    <div>
                        <span class="task-id">${task.id}</span>
                        <span class="task-model">${task.assigned_model || ''}</span>
                    </div>
                    <span>${task.description ? task.description.substring(0, 40) + '...' : ''}</span>
                </div>
                <div class="task-progress-bar-container">
                    <div class="task-progress-bar ${statusClass}" style="width: ${progress}%;"></div>
                </div>
                ${subtasksHtml}
            </div>
        `;
    }).join('');
}

function toggleSubtasks(taskId) {
    const subtasksEl = document.getElementById(`subtasks-${taskId}`);
    if (subtasksEl) {
        subtasksEl.classList.toggle('expanded');
    }
}

function updateWorkingOn(taskId, model, status) {
    const workingOn = document.getElementById('workingOnText');
    if (!workingOn) return;
    
    if (status === 'running') {
        workingOn.textContent = `Working on ${taskId} using ${model || 'auto-selected model'}...`;
    } else if (status === 'completed') {
        workingOn.textContent = `Completed ${taskId}`;
    } else {
        workingOn.textContent = 'Waiting...';
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendMessageBtn');
    const message = input.value.trim();
    
    if (!message) return;
    
    const model = document.getElementById('chatModelSelect')?.value;
    
    // If no project selected, create a new project
    if (!currentProjectId) {
        await createNewProject(message, model);
        return;
    }
    
    // Add user message to UI immediately
    const messagesEl = document.getElementById('chatMessages');
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.innerHTML = `
        <div class="chat-message-avatar"><i class="fas fa-user"></i></div>
        <div class="chat-message-content">${escapeHtml(message)}</div>
    `;
    messagesEl.appendChild(userMsg);
    
    // Add thinking indicator
    const thinkingMsg = document.createElement('div');
    thinkingMsg.className = 'chat-message assistant';
    thinkingMsg.id = 'thinkingMsg';
    thinkingMsg.innerHTML = `
        <div class="chat-message-avatar"><i class="fas fa-robot"></i></div>
        <div class="chat-message-content"><i class="fas fa-spinner fa-spin"></i> Thinking...</div>
    `;
    messagesEl.appendChild(thinkingMsg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    
    // Disable input
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    
    try {
        const res = await fetch(`/api/project/${currentProjectId}/message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await safeJsonParse(res);
        
        // Remove thinking message
        document.getElementById('thinkingMsg')?.remove();
        
        // Handle different response statuses
        if (!data) {
            addActivity('No response from server', 'error');
        } else if (data.status === 'ok' && data.response) {
            // Add AI response
            const aiMsg = document.createElement('div');
            aiMsg.className = 'chat-message assistant';
            aiMsg.innerHTML = `
                <div class="chat-message-avatar"><i class="fas fa-robot"></i></div>
                <div class="chat-message-content">${formatMessageContent(data.response)}</div>
            `;
            messagesEl.appendChild(aiMsg);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (data.status === 'partial' || data.status === 'error') {
            // Handle partial/error responses - show raw output for review
            const errorMsg = document.createElement('div');
            errorMsg.className = 'chat-message assistant';
            const content = data.raw_output || data.error || 'Unknown error';
            errorMsg.innerHTML = `
                <div class="chat-message-avatar"><i class="fas fa-exclamation-triangle"></i></div>
                <div class="chat-message-content" style="background: #fff3cd;">
                    <strong>Partial Response:</strong><br>
                    ${formatMessageContent(content)}
                </div>
            `;
            messagesEl.appendChild(errorMsg);
            messagesEl.scrollTop = messagesEl.scrollHeight;
            addActivity('Received partial response - review output above', 'warning');
        } else if (data.error) {
            addActivity(`Error: ${data.error}`, 'error');
        } else {
            addActivity('Unknown response format', 'error');
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        addActivity(`Error: ${error}`, 'error');
    } finally {
        input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.value = '';
        input.focus();
    }
}

async function createNewProject(message, model) {
    const sendBtn = document.getElementById('sendMessageBtn');
    const input = document.getElementById('chatInput');
    const systemPrompt = document.getElementById('systemPromptInput')?.value || '';
    
    addActivity('Analyzing goal and creating project plan...');
    
    // Show thinking in chat
    const messagesEl = document.getElementById('chatMessages');
    const thinkingMsg = document.createElement('div');
    thinkingMsg.className = 'chat-message assistant';
    thinkingMsg.id = 'thinkingMsg';
    thinkingMsg.innerHTML = `
        <div class="chat-message-avatar"><i class="fas fa-robot"></i></div>
        <div class="chat-message-content"><i class="fas fa-spinner fa-spin"></i> Analyzing goal and planning tasks...</div>
    `;
    messagesEl.innerHTML = '';
    messagesEl.appendChild(thinkingMsg);
    
    input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    
    try {
        const res = await fetch('/api/new-project', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ goal: message, model, system_prompt: systemPrompt, auto_run: true })
        });
        const data = await safeJsonParse(res);
        
        document.getElementById('thinkingMsg')?.remove();
        
        if (!data) {
            messagesEl.innerHTML = `
                <div class="chat-empty">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>No response from server</p>
                </div>
            `;
            addActivity('No response from server', 'error');
        } else if (data.error) {
            messagesEl.innerHTML = `
                <div class="chat-empty">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>Error: ${data.error}</p>
                </div>
            `;
            addActivity('Error: ' + data.error, 'error');
        } else if (data.needs_context) {
            // Show the follow-up question to the user
            const aiMsg = document.createElement('div');
            aiMsg.className = 'chat-message assistant';
            aiMsg.innerHTML = `
                <div class="chat-message-avatar"><i class="fas fa-robot"></i></div>
                <div class="chat-message-content">
                    <p>${data.follow_up_question || 'I need more context to create a proper plan. Could you provide more details?'}</p>
                    <div class="follow-up-prompt" style="margin-top: 1rem;">
                        <input type="text" class="input" id="followUpInput" placeholder="Provide more details..." style="width: 70%;">
                        <button class="btn btn-primary" onclick="submitFollowUp()">Submit</button>
                    </div>
                </div>
            `;
            messagesEl.innerHTML = '';
            messagesEl.appendChild(aiMsg);
            messagesEl.scrollTop = messagesEl.scrollHeight;
            
            // Store the original message for follow-up
            window.pendingGoal = message;
            window.pendingModel = model;
            
            // Allow pressing Enter to submit
            document.getElementById('followUpInput')?.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') submitFollowUp();
            });
            
        } else if (data.project_id) {
            currentProjectId = data.project_id;
            
            // Show the initial plan from the assistant's response
            if (data.conversation) {
                renderConversation(data.conversation);
            }
            
            // Show tasks in the task queue
            if (data.tasks && data.tasks.length > 0) {
                const tasks = data.tasks.map((t, i) => ({
                    id: t.id,
                    description: t.description,
                    assigned_model: t.assigned_model_id,
                    status: 'pending'
                }));
                renderTasks(tasks);
            }
            
            addActivity(`Project created: ${data.project_id} with ${data.tasks?.length || 0} tasks. Starting execution...`);
            
            // Show warnings if any
            if (data.warnings && data.warnings.length > 0) {
                showStatusBanner(data.warnings[0], 'warning');
            } else {
                showStatusBanner('Project created! Tasks are executing...', 'success');
            }
            
            // Poll for task completion status
            pollProjectStatus(data.project_id);
            
            // Refresh projects list
            loadSidebarProjects();
            
        } else {
            messagesEl.innerHTML = `
                <div class="chat-empty">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>Unexpected response</p>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Error creating project:', error);
        messagesEl.innerHTML = `
            <div class="chat-empty">
                <i class="fas fa-exclamation-circle"></i>
                <p>Error: ${error}</p>
            </div>
        `;
        addActivity('Error: ' + error, 'error');
    } finally {
        input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        input.value = '';
        input.focus();
    }
}

// Submit follow-up with more context
async function submitFollowUp() {
    const followUpInput = document.getElementById('followUpInput');
    if (!followUpInput) return;
    
    const additionalContext = followUpInput.value.trim();
    if (!additionalContext) {
        addActivity('Please provide more details', 'warning');
        return;
    }
    
    // Combine original goal with additional context
    const enhancedGoal = `${window.pendingGoal}\n\nAdditional context: ${additionalContext}`;
    
    // Clear the follow-up prompt and show thinking
    const messagesEl = document.getElementById('chatMessages');
    const thinkingMsg = document.createElement('div');
    thinkingMsg.className = 'chat-message assistant';
    thinkingMsg.id = 'thinkingMsg';
    thinkingMsg.innerHTML = `
        <div class="chat-message-avatar"><i class="fas fa-robot"></i></div>
        <div class="chat-message-content"><i class="fas fa-spinner fa-spin"></i> Creating plan with your additional context...</div>
    `;
    messagesEl.innerHTML = '';
    messagesEl.appendChild(thinkingMsg);
    
    // Disable input
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendMessageBtn');
    if (input) input.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    
    try {
        const res = await fetch('/api/new-project', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                goal: enhancedGoal, 
                model: window.pendingModel, 
                system_prompt: document.getElementById('systemPromptInput')?.value || '',
                auto_run: true 
            })
        });
        const data = await safeJsonParse(res);
        
        document.getElementById('thinkingMsg')?.remove();
        
        if (data.error) {
            messagesEl.innerHTML = `
                <div class="chat-empty">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>Error: ${data.error}</p>
                </div>
            `;
            addActivity('Error: ' + data.error, 'error');
        } else if (data.project_id) {
            currentProjectId = data.project_id;
            
            if (data.conversation) {
                renderConversation(data.conversation);
            }
            
            if (data.tasks && data.tasks.length > 0) {
                const tasks = data.tasks.map((t, i) => ({
                    id: t.id,
                    description: t.description,
                    assigned_model: t.assigned_model_id,
                    status: 'pending'
                }));
                renderTasks(tasks);
            }
            
            addActivity(`Project created with additional context: ${data.project_id}`);
            showStatusBanner('Project created! Tasks are executing...', 'success');
            pollProjectStatus(data.project_id);
            loadSidebarProjects();
        }
        
    } catch (error) {
        console.error('Error submitting follow-up:', error);
        addActivity('Error: ' + error, 'error');
    } finally {
        if (input) input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        window.pendingGoal = null;
        window.pendingModel = null;
    }
}

window.submitFollowUp = submitFollowUp;
window.toggleSubtasks = toggleSubtasks;

// Poll for project status updates
function pollProjectStatus(projectId) {
    const pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/project/${projectId}`);
            const data = await safeJsonParse(res);
            
            if (data.error) {
                clearInterval(pollInterval);
                return;
            }
            
            // Update task status display with progress
            if (data.tasks && data.tasks.length > 0) {
                const tasksWithStatus = data.tasks.map(t => ({
                    id: t.id,
                    description: t.description,
                    output: t.output,
                    files: t.files,
                    assigned_model: t.assigned_model,
                    status: t.output ? 'completed' : (t.status === 'running' ? 'running' : 'pending'),
                    subtasks: t.subtasks || []
                }));
                
                renderTasks(tasksWithStatus);
                showProgress(tasksWithStatus);
                
                // Update "working on" indicator
                const runningTask = tasksWithStatus.find(t => t.status === 'running');
                if (runningTask) {
                    updateWorkingOn(runningTask.id, runningTask.assigned_model, 'running');
                } else {
                    const nextTask = tasksWithStatus.find(t => t.status === 'pending');
                    if (nextTask) {
                        updateWorkingOn(nextTask.id, nextTask.assigned_model, 'pending');
                    }
                }
            }
            
            // Update conversation - append any new messages
            if (data.conversation && data.conversation.length > 0) {
                renderConversation(data.conversation, true);
            }
            
            // Check if project is completed
            const config = data.config;
            if (config && config.status === 'completed') {
                clearInterval(pollInterval);
                showStatusBanner('All tasks completed! Final deliverable assembled.', 'success');
                addActivity('Project completed!');
                loadSidebarProjects();
                updateWorkingOn('', '', 'completed');
                
                // Show final delivery panel
                if (config.final_delivery_path) {
                    showFinalDeliveryPanel(config, data.tasks);
                }
            } else if (config && config.status === 'error') {
                clearInterval(pollInterval);
                showStatusBanner('Project failed: ' + (config.error || 'Unknown error'), 'error');
                addActivity('Project failed: ' + (config.error || 'Unknown error'), 'error');
            }
            
        } catch (error) {
            console.error('Error polling project status:', error);
        }
    }, 3000);  // Poll every 3 seconds
    
    // Stop polling after 5 minutes max
    setTimeout(() => clearInterval(pollInterval), 300000);
}

function showFinalDeliveryPanel(config, tasks) {
    const panel = document.getElementById('finalDeliveryPanel');
    const content = document.getElementById('finalDeliveryContent');
    
    if (!panel || !content) return;
    
    let taskHtml = '';
    if (tasks && tasks.length > 0) {
        tasks.forEach(t => {
            const status = t.output ? '✓' : '○';
            taskHtml += `
                <div class="final-delivery-task">
                    <h5>[${status}] ${t.id}: ${t.description || 'No description'}</h5>
                    <span class="model-tag"><i class="fas fa-microchip"></i> ${t.assigned_model || 'Not assigned'}</span>
                    ${t.files && t.files.length > 0 ? `<div style="margin-top: 0.5rem;"><strong>Generated Files:</strong> ${t.files.map(f => f.name).join(', ')}</div>` : ''}
                </div>
            `;
        });
    }
    
    content.innerHTML = `
        <div style="margin-bottom: 1rem;">
            <strong>Goal:</strong> ${config.high_level_goal || 'N/A'}
        </div>
        <div style="margin-bottom: 1rem;">
            <strong>Final Report:</strong> <a href="/api/project/${config.project_name}/download-report" target="_blank">Open Final Report</a>
        </div>
        <div>
            <strong>Tasks:</strong>
            ${taskHtml}
        </div>
    `;
    
    panel.style.display = 'block';
}

// Dashboard
async function loadDashboard() {
    try {
        const [statusRes, modelsRes, projectsRes] = await Promise.all([
            fetch('/api/status'),
            fetch('/api/models'),
            fetch('/api/projects')
        ]);
        
        const status = await safeJsonParse(statusRes);
        const modelsData = await safeJsonParse(modelsRes);
        const projectsData = await safeJsonParse(projectsRes);
        
        const models = modelsData.models || [];
        const projects = projectsData.projects || [];
        _lastModelCount = models.length;   // Cache for checkLMStudioConnection

        // Update stats
        document.getElementById('modelCount').textContent = models.length;
        
        // Calculate tasks from projects
        let totalTasks = 0;
        let completedTasks = 0;
        let runningTasks = 0;
        
        projects.forEach(p => {
            const tasks = p.tasks || [];
            totalTasks += tasks.length;
            tasks.forEach(t => {
                if (t.status === 'completed') completedTasks++;
                else if (t.status === 'running') runningTasks++;
            });
        });
        
        document.getElementById('taskCount').textContent = totalTasks;
        document.getElementById('completedCount').textContent = completedTasks;
        document.getElementById('runningCount').textContent = runningTasks;

        // Update LM Studio status bar
        updateLMStatusBar(models.length > 0, models.length, status.host || '');

        // Populate model dropdown in quick prompt (if present)
        const qpModelSel = document.getElementById('quickPromptModel');
        if (qpModelSel && models.length > 0) {
            const current = qpModelSel.value;
            qpModelSel.innerHTML = models.map(m =>
                `<option value="${m.id}" ${m.id === current ? 'selected' : ''}>${m.name || m.id}</option>`
            ).join('');
        }

        // Show onboarding card if no projects yet
        const onboardingContainer = document.getElementById('onboardingContainer');
        if (onboardingContainer) {
            if (projects.length === 0) {
                onboardingContainer.style.display = 'block';
            } else {
                onboardingContainer.style.display = 'none';
            }
        }

        addActivity(`Dashboard refreshed — ${models.length} models, ${projects.length} projects`);
        
    } catch (error) {
        console.error('Error loading dashboard:', error);
        addActivity('Error loading dashboard', 'error');
        updateLMStatusBar(false, 0, '');
    }
}

async function fetchOutputs() {
    const outputs = [];
    try {
        const res = await fetch('/api/output/t1');
        const data = await safeJsonParse(res);
        if (data.output) outputs.push('t1');
    } catch {}
    try {
        const res = await fetch('/api/output/t2');
        const data = await safeJsonParse(res);
        if (data.output) outputs.push('t2');
    } catch {}
    return outputs;
}

function addActivity(message, type = 'info') {
    const time = new Date().toLocaleTimeString();
    activityLog.unshift({ message, time, type });
    if (activityLog.length > 10) activityLog.pop();
    renderActivity();
}

function renderActivity() {
    const list = document.getElementById('activityList');
    list.innerHTML = activityLog.map(item => `
        <div class="activity-item">
            <i class="fas fa-${item.type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${item.message}</span>
            <span class="time">${item.time}</span>
        </div>
    `).join('');
}

// Workflow
async function loadWorkflow() {
    const treeContainer = document.getElementById('workflowTree');
    treeContainer.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const res = await fetch('/api/workflow');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            treeContainer.innerHTML = '<p>Error loading workflow: ' + data.error + '</p>';
            return;
        }
        
        // Use projects data instead of workflow
        const projects = data.projects || [];
        
        if (projects.length === 0) {
            treeContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <p>No projects yet.</p>
                    <button class="btn btn-small btn-primary" onclick="switchView('prompt')">
                        <i class="fas fa-plus"></i> Create your first project
                    </button>
                </div>`;
            return;
        }
        
        treeContainer.innerHTML = projects.map(project => {
            const tasks = project.tasks || [];
            const statusClass = project.status || 'pending';
            
            return `
                <div class="project-tree-item ${statusClass}">
                    <div class="project-tree-header" onclick="toggleProjectTasks('${project.id}')">
                        <span class="tree-toggle"></span>
                        <div class="tree-content">
                            <div class="tree-label">
                                <i class="fas fa-folder"></i> ${project.name}
                                <span class="status-badge ${statusClass}">${project.status}</span>
                            </div>
                            <div class="tree-description">${project.goal || project.description || 'No goal'}</div>
                        </div>
                    </div>
                    <div class="tree-children expanded" id="workflow-tasks-${project.id}">
                        ${tasks.map(task => `
                            <div class="tree-node">
                                <div class="tree-item task ${task.status || 'pending'}">
                                    <span class="tree-toggle" style="visibility:hidden"></span>
                                    <div class="tree-content">
                                        <div class="tree-label">
                                            <i class="fas fa-tasks"></i> ${task.id}
                                            <span class="status-badge ${task.status || 'pending'}">${task.status || 'pending'}</span>
                                        </div>
                                        <div class="tree-description">${task.description || ''}</div>
                                        <div class="tree-model"><i class="fas fa-microchip"></i> ${task.assigned_model || 'Not assigned'}</div>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }).join('');
        
        // Add click handlers for toggles
        document.querySelectorAll('.tree-toggle').forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.stopPropagation();
                const parent = toggle.closest('.project-tree-item, .tree-node');
                const children = parent.querySelector('.tree-children');
                if (children) {
                    children.classList.toggle('expanded');
                    toggle.classList.toggle('expanded');
                }
            });
        });
    } catch (error) {
        treeContainer.innerHTML = '<p>Error loading workflow</p>';
    }
}

function renderWorkflowTree(node, isRoot = false) {
    const hasChildren = node.children && node.children.length > 0;
    const statusClass = node.status || '';
    
    let html = `
        <div class="tree-node ${isRoot ? 'tree-root' : ''}">
            <div class="tree-item ${node.type} ${statusClass}" data-id="${node.id}">
                ${hasChildren ? '<span class="tree-toggle"></span>' : '<span class="tree-toggle" style="visibility:hidden"></span>'}
                <div class="tree-content">
                    <div class="tree-label">${node.name}</div>
                    <div class="tree-description">${node.description || ''}</div>
                </div>
            </div>
    `;
    
    if (hasChildren) {
        html += `<div class="tree-children">`;
        for (const child of node.children) {
            html += renderWorkflowTree(child, false);
        }
        html += `</div>`;
    }
    
    html += `</div>`;
    return html;
}

function setupTreeToggle() {
    document.querySelectorAll('.tree-toggle').forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            const item = toggle.closest('.tree-item');
            const children = item.nextElementSibling;
            if (children && children.classList.contains('tree-children')) {
                children.classList.toggle('expanded');
                toggle.classList.toggle('expanded');
            }
        });
    });
}

// Decomposition Lab
async function loadDecomposition() {
    await Promise.all([
        loadTemplates(),
        loadKnobs(),
        loadDodDorCriteria(),
        loadDecompositionHistory(),
        loadAdrList()
    ]);
    setupDecompositionEventListeners();
}

async function loadTemplates() {
    const grid = document.getElementById('templateGrid');
    grid.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const agentFilter = document.getElementById('templateAgentFilter')?.value || '';
        const dodFilter = document.getElementById('templateDodFilter')?.value || '';
        
        let url = '/api/decomposition/templates?';
        if (agentFilter) url += `agent_type=${agentFilter}&`;
        if (dodFilter) url += `dod_level=${dodFilter}&`;
        
        const res = await fetch(url);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            grid.innerHTML = `<p>Error loading templates: ${data.error}</p>`;
            return;
        }
        
        if (data.templates.length === 0) {
            grid.innerHTML = '<p class="text-muted">No templates found</p>';
            return;
        }
        
        grid.innerHTML = data.templates.map(t => `
            <div class="template-card">
                <div class="template-header">
                    <h4>${t.name}</h4>
                    <span class="badge badge-${t.agent_type}">${t.agent_type}</span>
                </div>
                <p class="template-desc">${t.description}</p>
                <div class="template-meta">
                    <span><i class="fas fa-layer-group"></i> DoD: ${t.dod_level}</span>
                    <span><i class="fas fa-clock"></i> Effort: ${t.estimated_effort}</span>
                </div>
                <div class="template-keywords">
                    ${t.keywords.map(k => `<span class="keyword">${k}</span>`).join('')}
                </div>
            </div>
        `).join('');
    } catch (e) {
        grid.innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

async function loadSeedConfig() {
    const container = document.getElementById('seedConfig');
    if (!container) return;
    
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const res = await fetch('/api/decomposition/seed-config');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            container.innerHTML = `<p>Error: ${data.error}</p>`;
            return;
        }
        
        container.innerHTML = `
            <div class="seed-stats">
                <div class="seed-stat">
                    <span class="stat-label">Oracle</span>
                    <span class="stat-value">${(data.seed_mix.oracle * 100)}%</span>
                </div>
                <div class="seed-stat">
                    <span class="stat-label">Researcher</span>
                    <span class="stat-value">${(data.seed_mix.researcher * 100)}%</span>
                </div>
                <div class="seed-stat">
                    <span class="stat-label">Explorer</span>
                    <span class="stat-value">${(data.seed_mix.explorer * 100)}%</span>
                </div>
            </div>
            <div class="seed-details">
                <p><strong>Base seeds/cycle:</strong> ${data.seed_rate.base}</p>
                <p><strong>Max seeds/cycle:</strong> ${data.seed_rate.max}</p>
                <p><strong>Subtask cap:</strong> ${data.seed_rate.subtask_min}-${data.seed_rate.subtask_max}</p>
                <p><strong>Depth range:</strong> ${data.min_max_depth}-${data.max_max_depth} (default: ${data.default_max_depth})</p>
            </div>
        `;
    } catch (e) {
        container.innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

async function loadDodDorCriteria() {
    const dodLevel = document.getElementById('dodLevelSelect')?.value || 'Standard';
    const dorLevel = document.getElementById('dorLevelSelect')?.value || 'Standard';
    
    try {
        const [dodRes, dorRes] = await Promise.all([
            fetch(`/api/decomposition/dod-dor?level=${dodLevel}`),
            fetch(`/api/decomposition/dod-dor?level=${dorLevel}`)
        ]);
        
        const dodData = await safeJsonParse(dodRes);
        const dorData = await safeJsonParse(dorRes);
        
        const dodCriteria = document.getElementById('dodCriteria');
        const dorCriteria = document.getElementById('dorCriteria');
        
        if (dodCriteria) {
            dodCriteria.innerHTML = dodData.dod_criteria?.map(c => `<li>${c}</li>`).join('') || '';
        }
        if (dorCriteria) {
            dorCriteria.innerHTML = dorData.dor_criteria?.map(c => `<li>${c}</li>`).join('') || '';
        }
    } catch (e) {
        console.error('Error loading DoD/DoR criteria:', e);
    }
}

async function loadDecompositionHistory() {
    const list = document.getElementById('historyList');
    if (!list) return;
    
    list.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const planFilter = document.getElementById('historyPlanFilter')?.value || '';
        let url = '/api/decomposition/history?';
        if (planFilter) url += `plan_id=${planFilter}`;
        
        const res = await fetch(url);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            list.innerHTML = `<p>Error: ${data.error}</p>`;
            return;
        }
        
        if (data.history.length === 0) {
            list.innerHTML = '<p class="text-muted">No decomposition history yet</p>';
            return;
        }
        
        list.innerHTML = data.history.map(h => `
            <div class="history-item">
                <div class="history-header">
                    <span class="history-id">${h.event_id}</span>
                    <span class="history-time">${new Date(h.timestamp).toLocaleString()}</span>
                </div>
                <div class="history-details">
                    <span>Task: ${h.task_id}</span>
                    <span>Depth: ${h.depth}</span>
                    <span>Subtasks: ${h.subtasks_created}</span>
                </div>
                <div class="history-seeds">
                    Seeds: ${h.seeds_used.map(s => `<span class="seed-badge">${s}</span>`).join('')}
                </div>
            </div>
        `).join('');
    } catch (e) {
        list.innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

async function loadAdrList() {
    const list = document.getElementById('adrList');
    if (!list) return;
    
    list.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const res = await fetch('/api/adr');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            list.innerHTML = `<p>Error: ${data.error}</p>`;
            return;
        }
        
        if (data.adrs.length === 0) {
            list.innerHTML = '<p class="text-muted">No ADR records yet</p>';
            return;
        }
        
        list.innerHTML = data.adrs.map(a => `
            <div class="adr-item" data-adr-id="${a.adr_id}">
                <div class="adr-header">
                    <span class="adr-id">${a.adr_id}</span>
                    <span class="adr-status ${a.status}">${a.status}</span>
                    <span class="adr-category">${a.category}</span>
                </div>
                <h4 class="adr-title">${a.title}</h4>
                <p class="adr-context">${a.context.substring(0, 100)}${a.context.length > 100 ? '...' : ''}</p>
            </div>
        `).join('');
    } catch (e) {
        list.innerHTML = `<p>Error: ${e.message}</p>`;
    }
}

async function decomposeWithAgent() {
    const prompt = document.getElementById('decomposePrompt').value;
    const planId = document.getElementById('decomposePlanId').value || 'default';
    
    if (!prompt.trim()) {
        showStatusBanner('Please enter a task prompt', 'error');
        return;
    }
    
    const btn = document.getElementById('decomposeAgentBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div> Decomposing...';
    }
    
    try {
        const res = await fetch('/api/decomposition/decompose-agent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plan_id: planId,
                prompt: prompt
            })
        });
        
        const data = await safeJsonParse(res);
        
        if (data.error) {
            showStatusBanner(data.error, 'error');
            return;
        }
        
        document.getElementById('subtaskCount').textContent = data.count;
        
        const tree = document.getElementById('subtaskTree');
        if (data.subtasks.length === 0) {
            tree.innerHTML = '<p class="text-muted">No subtasks generated</p>';
        } else {
            tree.innerHTML = renderSubtaskTree(data.subtasks, data.plan_id);
        }
        
        showStatusBanner(`Generated ${data.count} subtasks (depth: ${data.depth})`, 'success');
        
        loadDecompositionHistory();
    } catch (e) {
        showStatusBanner(e.message, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-robot"></i> Decompose with Agent';
        }
    }
}

function renderSubtaskTree(subtasks, planId) {
    const byParent = {};
    subtasks.forEach(st => {
        if (!byParent[st.parent_id]) byParent[st.parent_id] = [];
        byParent[st.parent_id].push(st);
    });
    
    function renderNode(parentId, depth = 0) {
        const children = byParent[parentId] || [];
        if (children.length === 0 && depth > 0) return '';
        
        return children.map(st => `
            <div class="subtask-item" style="margin-left: ${depth * 20}px" data-subtask-id="${st.subtask_id}">
                <div class="subtask-header">
                    <span class="subtask-id">${st.subtask_id}</span>
                    <span class="badge badge-${st.agent_type}">${st.agent_type}</span>
                    <span class="depth-badge">D:${st.depth}</span>
                    <span class="subtask-dod">DoD: ${st.dod_level}</span>
                </div>
                <div class="subtask-desc">${st.description}</div>
                <div class="subtask-assigned">
                    ${st.assigned_agent ? `<span class="assigned-to">Assigned: ${st.assigned_agent}</span>` : '<span class="unassigned">Unassigned</span>'}
                </div>
                ${renderNode(st.subtask_id, depth + 1)}
            </div>
        `).join('');
    }
    
    return renderNode('root');
}

async function runAssignmentPass() {
    const planId = document.getElementById('decomposePlanId').value || 'default';
    
    const btn = document.getElementById('runAssignmentBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div> Assigning...';
    }
    
    try {
        const res = await fetch('/api/assignments/execute-pass', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plan_id: planId,
                auto_assign: true
            })
        });
        
        const data = await safeJsonParse(res);
        
        if (data.error) {
            showStatusBanner(data.error, 'error');
            return;
        }
        
        showStatusBanner(`Assigned ${data.assigned} subtasks, ${data.unassigned} unassigned`, 'success');
        
        loadSubtaskTree(planId);
    } catch (e) {
        showStatusBanner(e.message, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-users"></i> Run Assignment Pass';
        }
    }
}

async function loadSubtaskTree(planId) {
    try {
        const res = await fetch(`/api/subtasks/${planId}/tree`);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            console.error(data.error);
            return;
        }
        
        const tree = document.getElementById('subtaskTree');
        if (data.subtasks && data.subtasks.length > 0) {
            tree.innerHTML = renderSubtaskTree(data.subtasks, planId);
            document.getElementById('subtaskCount').textContent = data.total;
        }
    } catch (e) {
        console.error('Error loading subtask tree:', e);
    }
}

async function loadKnobs() {
    try {
        const res = await fetch('/api/decomposition/knobs');
        const data = await safeJsonParse(res);
        
        const container = document.getElementById('seedConfig');
        if (container && !data.error) {
            container.innerHTML = `
                <div class="seed-stats">
                    <div class="seed-stat">
                        <span class="stat-label">Oracle</span>
                        <span class="stat-value">${(data.seed_mix.oracle * 100)}%</span>
                    </div>
                    <div class="seed-stat">
                        <span class="stat-label">Researcher</span>
                        <span class="stat-value">${(data.seed_mix.researcher * 100)}%</span>
                    </div>
                    <div class="seed-stat">
                        <span class="stat-label">Explorer</span>
                        <span class="stat-value">${(data.seed_mix.explorer * 100)}%</span>
                    </div>
                </div>
                <div class="seed-details">
                    <p><strong>Base seeds/cycle:</strong> ${data.seed_rate.base}</p>
                    <p><strong>Max seeds/cycle:</strong> ${data.seed_rate.max}</p>
                    <p><strong>Subtask cap:</strong> ${data.seed_rate.subtask_min}-${data.seed_rate.subtask_max}</p>
                    <p><strong>Depth range:</strong> ${data.min_max_depth}-${data.max_max_depth} (default: ${data.default_max_depth})</p>
                    <p><strong>Effort threshold:</strong> ${data.recursion_knobs.est_effort_threshold}</p>
                </div>
            `;
        }
    } catch (e) {
        console.error('Error loading knobs:', e);
    }
}

function setupDecompositionEventListeners() {
    // Tab switching
    document.querySelectorAll('.decomposition-container .tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.decomposition-container .tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.decomposition-container .tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
        });
    });
    
    // Template filters
    document.getElementById('templateAgentFilter')?.addEventListener('change', loadTemplates);
    document.getElementById('templateDodFilter')?.addEventListener('change', loadTemplates);
    
    // DoD/DoR level selectors
    document.getElementById('dodLevelSelect')?.addEventListener('change', loadDodDorCriteria);
    document.getElementById('dorLevelSelect')?.addEventListener('change', loadDodDorCriteria);
    
    // Decompose button
    document.getElementById('decomposeBtn')?.addEventListener('click', async () => {
        const prompt = document.getElementById('decomposePrompt').value;
        const depth = parseInt(document.getElementById('decomposeDepth').value) || 0;
        const maxDepth = parseInt(document.getElementById('decomposeMaxDepth').value) || 14;
        const planId = document.getElementById('decomposePlanId').value || 'default';
        
        if (!prompt.trim()) {
            showStatusBanner('Please enter a task prompt', 'error');
            return;
        }
        
        const btn = document.getElementById('decomposeBtn');
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div> Decomposing...';
        
        try {
            const res = await fetch('/api/decomposition/decompose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_prompt: prompt,
                    parent_task_id: 'root',
                    depth: depth,
                    max_depth: maxDepth,
                    plan_id: planId
                })
            });
            
            const data = await safeJsonParse(res);
            
            if (data.error) {
                showStatusBanner(data.error, 'error');
                return;
            }
            
            document.getElementById('subtaskCount').textContent = data.count;
            
            const tree = document.getElementById('subtaskTree');
            if (data.subtasks.length === 0) {
                tree.innerHTML = '<p class="text-muted">No subtasks generated (max depth reached)</p>';
            } else {
                tree.innerHTML = data.subtasks.map(st => `
                    <div class="subtask-item" style="margin-left: ${(st.depth || 1) * 20}px">
                        <div class="subtask-header">
                            <span class="subtask-id">${st.task_id}</span>
                            <span class="badge badge-${st.agent_type}">${st.agent_type}</span>
                            <span class="subtask-dod">DoD: ${st.dod_level}</span>
                        </div>
                        <div class="subtask-desc">${st.description}</div>
                        <div class="subtask-prompt">${st.prompt.substring(0, 80)}...</div>
                    </div>
                `).join('');
            }
            
            showStatusBanner(`Generated ${data.count} subtasks`, 'success');
            
            // Refresh history
            loadDecompositionHistory();
            
        } catch (e) {
            showStatusBanner(e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-sitemap"></i> Decompose';
        }
    });
    
    // History refresh
    document.getElementById('refreshHistoryBtn')?.addEventListener('click', loadDecompositionHistory);
    document.getElementById('historyPlanFilter')?.addEventListener('input', debounce(loadDecompositionHistory, 300));
}

// Models
async function loadModels() {
    const grid = document.getElementById('modelsGrid');
    if (!grid) return;
    grid.innerHTML = '<div class="loading"><div class="spinner"></div><p style="margin-top:8px;color:var(--text-muted);font-size:0.8rem;">Loading models...</p></div>';

    // Wire the Refresh button to force-bypass the cache
    const refreshBtn = document.getElementById('refreshModels');
    if (refreshBtn && !refreshBtn._wired) {
        refreshBtn._wired = true;
        refreshBtn.addEventListener('click', async () => {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            await fetch('/api/models/refresh', { method: 'POST' });
            await loadModels();
            refreshBtn.disabled = false;
            refreshBtn.innerHTML = '<i class="fas fa-sync"></i> Refresh';
        });
    }

    try {
        const res = await fetch('/api/models');
        const data = await safeJsonParse(res);

        if (data.error) {
            grid.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-triangle"></i><p>Error: ${data.error}</p><button class="btn btn-small btn-primary" onclick="loadModels()">Retry</button></div>`;
            return;
        }

        const modelList = data.models || [];
        _lastModelCount = modelList.length;

        if (modelList.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-microchip"></i>
                    <p>No models found. Make sure LM Studio is running and a model is loaded.</p>
                    <button class="btn btn-small btn-primary" onclick="document.getElementById('discoverBtn').click()">
                        <i class="fas fa-compass"></i> Discover Models
                    </button>
                </div>`;
            return;
        }

        grid.innerHTML = modelList.map(model => `
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-name" title="${model.id || model.name}">${model.name || model.id}</span>
                    ${model.version ? `<span class="model-badge">${model.version}</span>` : ''}
                </div>
                <div class="model-info">
                    ${model.memory_gb ? `<span><i class="fas fa-memory"></i> ${model.memory_gb} GB</span>` : ''}
                    ${model.context_len ? `<span><i class="fas fa-align-left"></i> ${model.context_len.toLocaleString()} ctx</span>` : ''}
                    ${(model.capabilities || []).length ? `<span><i class="fas fa-cogs"></i> ${model.capabilities.join(', ')}</span>` : ''}
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('loadModels error:', error);
        grid.innerHTML = `<div class="empty-state"><i class="fas fa-exclamation-triangle"></i><p>Error loading models: ${error.message}</p></div>`;
    }
}

// Tasks - Shows all projects and their tasks
async function loadTasks() {
    let list = document.getElementById('queueTasksList');
    if (!list) {
        // Try alternate selectors
        list = document.querySelector('#tasksView #tasksList');
    }
    if (!list) return;
    list.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const res = await fetch('/api/projects');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            list.innerHTML = '<p>Error loading projects: ' + data.error + '</p>';
            return;
        }
        
        const projects = data.projects || [];
        
        if (projects.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-tasks"></i>
                    <p>No tasks yet. Create a project first.</p>
                    <button class="btn btn-small btn-primary" onclick="switchView('prompt')">
                        <i class="fas fa-plus"></i> New Project
                    </button>
                </div>`;
            return;
        }
        
        list.innerHTML = projects.map(project => {
            const tasks = project.tasks || [];
            const statusClass = project.status === 'completed' ? 'completed' : 
                               project.status === 'running' ? 'running' : 'pending';
            
            return `
                <div class="project-section">
                    <div class="project-header" onclick="toggleProjectTasks('${project.id}')">
                        <div class="project-status ${statusClass}"></div>
                        <div class="project-info">
                            <span class="project-name">${project.name}</span>
                            <span class="project-goal">${project.goal || project.description || 'No goal'}</span>
                        </div>
                        <div class="project-meta">
                            <span class="task-count">${tasks.length} tasks</span>
                            <span class="status-badge ${statusClass}">${project.status}</span>
                        </div>
                        <i class="fas fa-chevron-down toggle-icon" id="toggle-${project.id}"></i>
                    </div>
                    <div class="project-tasks" id="tasks-${project.id}">
                        ${tasks.length === 0 ? '<p class="no-tasks">No tasks</p>' : tasks.map(task => {
                            const taskStatusClass = task.status || 'pending';
                            return `
                                <div class="task-item ${taskStatusClass}">
                                    <div class="task-status ${taskStatusClass}">
                                        ${taskStatusClass === 'completed' ? '<i class="fas fa-check"></i>' : 
                                          taskStatusClass === 'running' ? '<i class="fas fa-spinner fa-spin"></i>' : 
                                          '<i class="fas fa-clock"></i>'}
                                    </div>
                                    <div class="task-info">
                                        <span class="task-name">${task.id}</span>
                                        <span class="task-desc">${task.description || 'No description'}</span>
                                        <span class="task-model"><i class="fas fa-microchip"></i> ${task.assigned_model || 'Not assigned'}</span>
                                    </div>
                                    <div class="task-actions">
                                        ${taskStatusClass !== 'completed' ? `
                                            <button class="btn btn-small btn-secondary" onclick="runProjectTask('${project.id}', '${task.id}')">
                                                <i class="fas fa-play"></i> Run
                                            </button>
                                        ` : ''}
                                        <button class="btn btn-small btn-secondary" onclick="viewProjectTaskOutput('${project.id}', '${task.id}')">
                                            <i class="fas fa-eye"></i> View
                                        </button>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        list.innerHTML = '<p>Error loading projects</p>';
    }
}

// Toggle project tasks visibility
function toggleProjectTasks(projectId) {
    const tasksEl = document.getElementById(`tasks-${projectId}`);
    const toggleEl = document.getElementById(`toggle-${projectId}`);
    if (tasksEl) {
        tasksEl.classList.toggle('expanded');
        if (toggleEl) {
            toggleEl.classList.toggle('expanded');
        }
    }
}

// Run a specific task in a project
async function runProjectTask(projectId, taskId) {
    addActivity(`Running task ${taskId} in project ${projectId}...`);
    // For now, just refresh to show status
    loadTasks();
}

// View task output from a project
async function viewProjectTaskOutput(projectId, taskId) {
    const outputEl = document.getElementById('codeOutput');
    if (!outputEl) {
        switchView('output');
        return;
    }
    
    outputEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const res = await fetch(`/api/project/${projectId}/task/${taskId}/output`);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            outputEl.textContent = 'Error loading output: ' + data.error;
            return;
        }
        
        if (data.output) {
            outputEl.textContent = data.output;
            
            // Show generated files if any
            if (data.files && data.files.length > 0) {
                outputEl.innerHTML += '\n\n--- Generated Files ---\n';
                data.files.forEach(f => {
                    outputEl.innerHTML += `\n${f.name}`;
                });
            }
        } else {
            outputEl.textContent = 'No output available for task ' + taskId;
        }
        
        // Switch to output view
        switchView('output');
    } catch (error) {
        outputEl.textContent = 'Error loading output';
    }
}

// Make functions global
window.toggleProjectTasks = toggleProjectTasks;
window.runProjectTask = runProjectTask;
window.viewProjectTaskOutput = viewProjectTaskOutput;

async function runTask(taskId) {
    addActivity(`Running task ${taskId}...`);
    try {
        const res = await fetch('/api/run-task', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId })
        });
        const data = await safeJsonParse(res);
        if (data.status === 'started') {
            addActivity(`Task ${taskId} started`);
            // Refresh after a delay
            setTimeout(loadTasks, 5000);
        } else {
            addActivity(`Error: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        addActivity(`Error running task: ${error}`, 'error');
    }
}

// Output
async function loadOutput() {
    const select = document.getElementById('outputTaskSelect');
    if (!select) return;
    
    try {
        // Load all tasks
        const res = await fetch('/api/all-tasks');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            console.error('Error loading tasks:', data.error);
            return;
        }
        
        const tasks = data.tasks || [];

        if (tasks.length === 0) {
            select.innerHTML = '<option value="">No tasks yet — create a project first</option>';
            const outputEl = document.getElementById('codeOutput');
            if (outputEl) outputEl.textContent = 'No task output yet. Create a project in "New Project" and run some tasks.';
            return;
        }

        // Populate select dropdown
        let currentSelection = select.value;
        select.innerHTML = '<option value="">Select a task...</option>';
        
        // Group by project
        let currentProject = '';
        tasks.forEach(t => {
            if (t.project_id !== currentProject) {
                if (currentProject !== '') {
                    // Add separator
                }
                const optgroup = document.createElement('optgroup');
                optgroup.label = t.project_name || t.project_id;
                select.appendChild(optgroup);
                currentProject = t.project_id;
            }
            
            const option = document.createElement('option');
            option.value = JSON.stringify({project_id: t.project_id, task_id: t.task_id});
            option.textContent = `${t.task_id}: ${t.description.substring(0, 50)}...`;
            option.dataset.projectId = t.project_id;
            option.dataset.taskId = t.task_id;
            select.appendChild(option);
        });
        
        // Restore selection if still valid
        if (currentSelection) {
            const found = Array.from(select.options).some(o => o.value === currentSelection);
            if (found) {
                select.value = currentSelection;
            }
        }
        
        // If there's a current project, select first task
        if (!select.value && tasks.length > 0) {
            const firstTask = tasks.find(t => t.project_id === currentProjectId) || tasks[0];
            if (firstTask) {
                select.value = JSON.stringify({project_id: firstTask.project_id, task_id: firstTask.task_id});
            }
        }
        
        if (select.value) {
            loadOutputForTask();
        }
        
    } catch (error) {
        console.error('Error loading output:', error);
    }
}

async function loadOutputForTask() {
    const select = document.getElementById('outputTaskSelect');
    const outputEl = document.getElementById('codeOutput');
    if (!select || !outputEl) return;
    
    const selectedValue = select.value;
    if (!selectedValue) {
        outputEl.textContent = 'Select a task to view its output';
        return;
    }
    
    try {
        let taskData;
        try {
            taskData = JSON.parse(selectedValue);
        } catch {
            // Fallback for old format
            taskData = {project_id: null, task_id: selectedValue};
        }
        
        let url = '';
        if (taskData.project_id) {
            url = `/api/project/${taskData.project_id}/task/${taskData.task_id}/output`;
        } else {
            url = `/api/output/${taskData.task_id}`;
        }
        
        outputEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
        
        const res = await fetch(url);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            outputEl.textContent = 'Error loading output: ' + data.error;
            return;
        }
        
        if (data.output) {
            outputEl.textContent = data.output;
            
            // Show generated files if any
            if (data.files && data.files.length > 0) {
                outputEl.innerHTML += '\n\n--- Generated Files ---\n';
                data.files.forEach(f => {
                    outputEl.innerHTML += `\n${f.name}`;
                });
            }
        } else {
            outputEl.textContent = 'No output available for this task';
        }
        
    } catch (error) {
        outputEl.textContent = 'Error loading output: ' + error;
    }
}

async function viewTaskOutput(taskId) {
    const outputEl = document.getElementById('codeOutput');
    if (!outputEl) return;
    
    outputEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const res = await fetch(`/api/output/${taskId}`);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            outputEl.textContent = 'Error loading output: ' + data.error;
            return;
        }
        
        outputEl.textContent = data.output || 'No output available';
    } catch (error) {
        outputEl.textContent = 'Error loading output';
    }
}

// Settings
async function loadSettings() {
    const hostInput = document.getElementById('hostInput');
    const configInput = document.getElementById('configInput');
    const apiTokenInput = document.getElementById('apiTokenInput');
    
    try {
        const res = await fetch('/api/status');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            console.error('Error loading settings:', data.error);
            return;
        }
        
        if (hostInput) hostInput.value = data.host || '';
        if (configInput) configInput.value = data.config_path || '';
        if (apiTokenInput) {
            // Only show if token is set (masked)
            apiTokenInput.value = data.api_token && data.api_token !== '' ? '********' : '';
            apiTokenInput.dataset.hasToken = data.api_token && data.api_token !== '' ? 'true' : 'false';
        }
        
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

// Agent Status Panel
async function refreshAgentStatus() {
    const grid = document.getElementById('agentStatusGrid');
    if (!grid) return;
    
    grid.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const res = await fetch('/api/agents/status');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            grid.innerHTML = '<p style="color: var(--error);">Error: ' + data.error + '</p>';
            return;
        }
        
        const agents = data.agents || [];
        
        if (agents.length === 0) {
            grid.innerHTML = '<p style="color: var(--text-secondary);">No agents initialized. Go to Agents view to initialize.</p>';
            return;
        }
        
        const agentIcons = {
            explorer: 'fa-compass',
            librarian: 'fa-book',
            oracle: 'fa-globe',
            'ui-planner': 'fa-palette',
            builder: 'fa-hammer',
            researcher: 'fa-search',
            evaluator: 'fa-check-circle',
            synthesizer: 'fa-brain'
        };
        
        grid.innerHTML = agents.map(agent => `
            <div class="agent-status-card">
                <div class="agent-status-icon ${agent.type}">
                    <i class="fas ${agentIcons[agent.type] || 'fa-robot'}"></i>
                </div>
                <div class="agent-status-info">
                    <div class="agent-status-name">${agent.name}</div>
                    <div class="agent-status-state">${agent.state || 'idle'}</div>
                </div>
                <div class="agent-status-dot ${agent.state === 'active' ? 'active' : 'idle'}"></div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading agent status:', error);
        grid.innerHTML = '<p style="color: var(--error);">Error loading agents</p>';
    }
}

// Shared Memory Panel
async function refreshSharedMemory() {
    const list = document.getElementById('sharedMemoryList');
    if (!list) return;
    
    try {
        const res = await fetch('/api/memory');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            list.innerHTML = '<p style="color: var(--error);">Error: ' + data.error + '</p>';
            return;
        }
        
        const memories = data.memories || [];
        
        if (memories.length === 0) {
            list.innerHTML = '<div class="empty-state"><i class="fas fa-memory"></i><p>No memories yet</p></div>';
            return;
        }
        
        const memoryIcons = {
            intent: 'fa-bullseye',
            discovery: 'fa-lightbulb',
            decision: 'fa-balance-scale',
            problem: 'fa-exclamation-triangle',
            solution: 'fa-check',
            pattern: 'fa-repeat',
            warning: 'fa-shield-alt'
        };
        
        list.innerHTML = memories.slice(0, 10).map(mem => `
            <div class="memory-item">
                <i class="fas ${memoryIcons[mem.type] || 'fa-memory'} memory-icon"></i>
                <div class="memory-content">
                    <div>${mem.content?.substring(0, 80) || ''}...</div>
                    <div class="memory-meta">
                        <span class="memory-tag">${mem.type}</span>
                        <span>${mem.agent}</span>
                    </div>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading shared memory:', error);
        list.innerHTML = '<p style="color: var(--error);">Error loading memory</p>';
    }
}

// Initialize Agents
async function initializeAgents() {
    addActivity('Initializing multi-agent orchestrator...');
    
    try {
        const res = await fetch('/api/agents/initialize', { method: 'POST' });
        const data = await safeJsonParse(res);
        
        if (data.error) {
            addActivity('Error initializing agents: ' + data.error, 'error');
            return;
        }
        
        addActivity('Agents initialized: ' + (data.agents?.join(', ') || 'None'));
        refreshAgentStatus();
        loadAgentsView();
        
    } catch (error) {
        console.error('Error initializing agents:', error);
        addActivity('Error initializing agents', 'error');
    }
}

// Load Agents View
async function loadAgentsView() {
    await refreshAgentStatus();
    await refreshSharedMemory();
    await loadActiveAgents();
    await loadAgentTasks();
    await loadDecisionPanel();
}

async function loadActiveAgents() {
    const list = document.getElementById('activeAgentsList');
    if (!list) return;
    
    try {
        const res = await fetch('/api/agents/active');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            list.innerHTML = '<p style="color: var(--error);">Error: ' + data.error + '</p>';
            return;
        }
        
        const agents = data.agents || [];
        
        if (agents.length === 0) {
            list.innerHTML = '<div class="empty-state"><i class="fas fa-robot"></i><p>No active agents</p></div>';
            return;
        }
        
        list.innerHTML = agents.map(agent => `
            <div class="active-agent-item">
                <div class="active-agent-avatar" style="background: ${agent.color || 'var(--accent)'}">
                    <i class="fas ${agent.icon || 'fa-robot'}"></i>
                </div>
                <div class="active-agent-info">
                    <div class="active-agent-name">${agent.name}</div>
                    <div class="active-agent-role">${agent.role}</div>
                </div>
                <div class="active-agent-tasks">${agent.tasks_completed || 0} tasks</div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading active agents:', error);
    }
}

async function loadAgentTasks() {
    const list = document.getElementById('agentTasksList');
    if (!list) return;
    
    try {
        const res = await fetch('/api/agents/tasks');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            list.innerHTML = '<p style="color: var(--error);">Error: ' + data.error + '</p>';
            return;
        }
        
        const tasks = data.tasks || [];
        
        if (tasks.length === 0) {
            list.innerHTML = '<div class="empty-state"><i class="fas fa-tasks"></i><p>No agent tasks</p></div>';
            return;
        }
        
        list.innerHTML = tasks.map(task => `
            <div class="agent-task-item">
                <span class="agent-task-status ${task.status}"></span>
                <div class="agent-task-info">${task.description}</div>
                <span class="agent-task-agent">${task.agent}</span>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading agent tasks:', error);
    }
}

async function loadDecisionPanel() {
    const panel = document.getElementById('decisionPanel');
    if (!panel) return;
    
    try {
        const res = await fetch('/api/decisions/pending');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            panel.innerHTML = '<p style="color: var(--error);">Error: ' + data.error + '</p>';
            return;
        }
        
        const decisions = data.decisions || [];
        
        if (decisions.length === 0) {
            panel.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.85rem;">No pending decisions</p>';
            return;
        }
        
        panel.innerHTML = decisions.map((dec, i) => `
            <div class="decision-item">
                <div class="decision-prompt">${dec.prompt}</div>
                <div class="decision-options">
                    ${(dec.options || []).map((opt, j) => `
                        <div class="decision-option" onclick="selectDecisionOption(${i}, ${j})">
                            <div class="decision-option-radio"></div>
                            <div class="decision-option-text">
                                <div class="decision-option-label">${opt.label}</div>
                                <div class="decision-option-pros">+ ${opt.pros || ''}</div>
                                <div class="decision-option-cons">- ${opt.cons || ''}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                <button class="btn btn-primary btn-small decision-submit" onclick="submitDecision(${i})">Submit Decision</button>
            </div>
        `).join('');
        
        window.pendingDecisions = decisions;
        
    } catch (error) {
        console.error('Error loading decision panel:', error);
    }
}

function selectDecisionOption(decisionIndex, optionIndex) {
    const options = document.querySelectorAll(`.decision-item:nth-child(${decisionIndex + 1}) .decision-option`);
    options.forEach((opt, i) => {
        opt.classList.toggle('selected', i === optionIndex);
    });
    window.selectedDecision = { decisionIndex, optionIndex };
}

async function submitDecision(decisionIndex) {
    const selection = window.selectedDecision;
    if (!selection || selection.decisionIndex !== decisionIndex) {
        alert('Please select an option');
        return;
    }
    
    const decision = window.pendingDecisions[decisionIndex];
    const option = decision.options[selection.optionIndex];
    
    try {
        const res = await fetch('/api/decisions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                decision_id: decision.id,
                choice: option.label
            })
        });
        const data = await safeJsonParse(res);
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        addActivity('Decision made: ' + option.label);
        loadDecisionPanel();
        
    } catch (error) {
        console.error('Error submitting decision:', error);
    }
}

window.refreshAgentStatus = refreshAgentStatus;
window.refreshSharedMemory = refreshSharedMemory;
window.initializeAgents = initializeAgents;
window.selectDecisionOption = selectDecisionOption;
window.submitDecision = submitDecision;

async function loadCredentials() {
    const listEl = document.getElementById('credentialsList');
    if (!listEl) return;
    
    try {
        const res = await fetch('/api/admin/credentials');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            listEl.innerHTML = '<p style="color: var(--error);">Error loading credentials</p>';
            return;
        }
        
        const health = data.health || {};
        const credentials = data.credentials || {};
        
        if (Object.keys(health).length === 0) {
            listEl.innerHTML = '<p style="color: var(--text-secondary);">No credentials configured. Add tokens below to enable live model search.</p>';
            return;
        }
        
        listEl.innerHTML = Object.entries(health).map(([source, info]) => `
            <div class="credential-item" style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; background: var(--bg-card); border-radius: var(--radius); margin-bottom: 0.5rem;">
                <div>
                    <strong>${source}</strong>
                    <span style="color: ${info.enabled ? 'var(--success)' : 'var(--error)'}; font-size: 0.8rem;">
                        ${info.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">
                        Last rotated: ${info.last_rotated ? info.last_rotated.split('T')[0] : 'Never'}
                        ${info.needs_rotation ? '<span style="color: var(--warning);"> - Rotation needed!</span>' : ''}
                    </div>
                </div>
                <button class="btn btn-small btn-secondary" onclick="deleteCredential('${source}')" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading credentials:', error);
        listEl.innerHTML = '<p style="color: var(--error);">Error loading credentials</p>';
    }
}

document.getElementById('saveCredentialBtn')?.addEventListener('click', async () => {
    const source = document.getElementById('credentialSourceSelect')?.value;
    const token = document.getElementById('credentialTokenInput')?.value;
    const rotation = document.getElementById('credentialRotationInput')?.value;
    
    if (!token) {
        alert('Please enter a token');
        return;
    }
    
    try {
        const res = await fetch(`/api/admin/credentials/${source}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                token: token,
                credential_type: 'bearer',
                rotation_days: parseInt(rotation) || 30
            })
        });
        
        const data = await safeJsonParse(res);
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        showStatus('Credential saved successfully');
        document.getElementById('credentialTokenInput').value = '';
        loadCredentials();
        
    } catch (error) {
        alert('Error saving credential');
    }
});

async function deleteCredential(source) {
    if (!confirm(`Delete credential for ${source}?`)) return;
    
    try {
        const res = await fetch(`/api/admin/credentials/${source}`, {
            method: 'DELETE'
        });
        
        const data = await safeJsonParse(res);
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        showStatus('Credential deleted');
        loadCredentials();
        
    } catch (error) {
        alert('Error deleting credential');
    }
}

document.getElementById('saveConfig')?.addEventListener('click', async () => {
    const host = document.getElementById('hostInput')?.value;
    const config = document.getElementById('configInput')?.value;
    const apiToken = document.getElementById('apiTokenInput')?.value;
    const btn = document.getElementById('saveConfig');
    
    // Show saving state
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    }
    
    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ host, config_path: config, api_token: apiToken })
        });
        const data = await safeJsonParse(res);
        
        if (data.status === 'updated') {
            addActivity('Settings saved successfully!');
            // Show success briefly then reload page
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } else {
            addActivity('Error saving settings: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        addActivity('Error saving settings: ' + error, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-save"></i> Save Settings';
        }
    }
});

// Model Configuration
async function loadModelConfig() {
    try {
        const res = await fetch('/api/model-config');
        const data = await safeJsonParse(res);
        
        if (data.error) {
            console.error('Error loading model config:', data.error);
            return;
        }
        
        const memoryInput = document.getElementById('memoryBudgetInput');
        const defaultModelsInput = document.getElementById('defaultModelsInput');
        const fallbackModelsInput = document.getElementById('fallbackModelsInput');
        const uncensoredModelsInput = document.getElementById('uncensoredModelsInput');
        
        if (memoryInput) memoryInput.value = data.memory_budget_gb || 32;
        if (defaultModelsInput) defaultModelsInput.value = (data.default_models || []).join(', ');
        if (fallbackModelsInput) fallbackModelsInput.value = (data.fallback_models || []).join(', ');
        if (uncensoredModelsInput) uncensoredModelsInput.value = (data.uncensored_fallback_models || []).join(', ');
        
        // Load status and models in parallel (models uses TTL cache, so fast)
        const [statusRes, modelsRes] = await Promise.all([
            fetch('/api/status'),
            fetch('/api/models')
        ]);
        const [statusData, modelsData] = await Promise.all([
            safeJsonParse(statusRes),
            safeJsonParse(modelsRes)
        ]);

        const activeModelDisplay = document.getElementById('activeModelDisplay');
        if (activeModelDisplay) {
            activeModelDisplay.innerHTML = statusData.active_model_id
                ? `<strong>${statusData.active_model_id}</strong>`
                : '<span style="color: var(--text-secondary);">No model active</span>';
        }

        const modelSwapSelect = document.getElementById('modelSwapSelect');
        if (modelSwapSelect && modelsData.models && modelsData.models.length > 0) {
            modelSwapSelect.innerHTML = '<option value="">Select a model...</option>' +
                modelsData.models.map(m => `<option value="${m.id}">${m.name || m.id}</option>`).join('');
        } else if (modelSwapSelect) {
            modelSwapSelect.innerHTML = '<option value="">No models — click Discover</option>';
        }
        
    } catch (error) {
        console.error('Error loading model config:', error);
    }
}

document.getElementById('saveModelConfig')?.addEventListener('click', async () => {
    const memoryBudget = document.getElementById('memoryBudgetInput')?.value;
    const defaultModels = document.getElementById('defaultModelsInput')?.value;
    const fallbackModels = document.getElementById('fallbackModelsInput')?.value;
    const uncensoredModels = document.getElementById('uncensoredModelsInput')?.value;
    const btn = document.getElementById('saveModelConfig');
    
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
    }
    
    try {
        const res = await fetch('/api/model-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                memory_budget_gb: parseInt(memoryBudget) || 32,
                default_models: defaultModels ? defaultModels.split(',').map(s => s.trim()).filter(s => s) : [],
                fallback_models: fallbackModels ? fallbackModels.split(',').map(s => s.trim()).filter(s => s) : [],
                uncensored_fallback_models: uncensoredModels ? uncensoredModels.split(',').map(s => s.trim()).filter(s => s) : []
            })
        });
        const data = await safeJsonParse(res);
        
        if (data.status === 'updated') {
            addActivity('Model configuration saved!');
        } else {
            addActivity('Error saving model config: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        addActivity('Error saving model config: ' + error, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-save"></i> Save Model Config';
        }
    }
});

document.getElementById('swapModelBtn')?.addEventListener('click', async () => {
    const modelSelect = document.getElementById('modelSwapSelect');
    const newModel = modelSelect?.value;
    
    if (!newModel) {
        addActivity('Please select a model to swap to', 'warning');
        return;
    }
    
    try {
        const res = await fetch('/api/swap-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: newModel })
        });
        const data = await safeJsonParse(res);
        
        if (data.status === 'swapped') {
            addActivity(`Swapped to model: ${newModel}`);
            loadModelConfig();
        } else {
            addActivity('Error swapping model: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        addActivity('Error swapping model: ' + error, 'error');
    }
});

// Archive View
async function loadArchive() {
    const list = document.getElementById('archiveList');
    if (!list) return;
    
    list.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    
    try {
        const searchQuery = document.getElementById('archiveSearch')?.value || '';
        const statusFilter = document.getElementById('archiveStatusFilter')?.value || '';
        
        let url = '/api/workflow?include_archived=true';
        if (searchQuery) url += `&search=${encodeURIComponent(searchQuery)}`;
        if (statusFilter) url += `&status=${encodeURIComponent(statusFilter)}`;
        
        const res = await fetch(url);
        const data = await safeJsonParse(res);
        
        if (data.error) {
            list.innerHTML = '<p>Error loading archive: ' + data.error + '</p>';
            return;
        }
        
        const projects = (data.projects || []).filter(p => p.archived);
        
        if (projects.length === 0) {
            list.innerHTML = '<div class="empty-state"><i class="fas fa-archive"></i><p>No archived projects</p></div>';
            return;
        }
        
        list.innerHTML = projects.map(project => `
            <div class="archive-item">
                <div class="archive-item-header">
                    <span class="archive-item-name">${project.name}</span>
                    <div class="archive-item-actions">
                        <button class="btn btn-small btn-secondary" onclick="renameProject('${project.id}')">
                            <i class="fas fa-edit"></i> Rename
                        </button>
                        <button class="btn btn-small btn-primary" onclick="unarchiveProject('${project.id}')">
                            <i class="fas fa-box-open"></i> Unarchive
                        </button>
                        <button class="btn btn-small btn-danger" onclick="deleteProject('${project.id}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="archive-item-meta">
                    <span><i class="fas fa-tag"></i> ${project.status}</span>
                    <span><i class="fas fa-microchip"></i> ${project.model || 'N/A'}</span>
                    <span>${project.tasks?.length || 0} tasks</span>
                </div>
                ${project.goal ? `<p style="margin-top: 0.5rem; color: var(--text-secondary);">${project.goal}</p>` : ''}
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading archive:', error);
        list.innerHTML = '<p>Error loading archive</p>';
    }
}

async function renameProject(projectId) {
    const newName = prompt('Enter new project name:');
    if (!newName) return;
    
    const newDescription = prompt('Enter new description (or leave empty):') || '';
    
    try {
        const res = await fetch(`/api/project/${projectId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName, description: newDescription })
        });
        const data = await safeJsonParse(res);
        
        if (data.status === 'renamed') {
            addActivity(`Project renamed to: ${newName}`);
            loadArchive();
            loadSidebarProjects();
        } else {
            addActivity('Error renaming project: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        addActivity('Error renaming project: ' + error, 'error');
    }
}

async function unarchiveProject(projectId) {
    if (!confirm('Unarchive this project?')) return;
    
    try {
        const res = await fetch(`/api/project/${projectId}/archive`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ archive: false })
        });
        const data = await safeJsonParse(res);
        
        if (data.status === 'unarchived') {
            addActivity('Project unarchived');
            loadArchive();
            loadSidebarProjects();
            loadWorkflow();
        } else {
            addActivity('Error unarchiving project: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        addActivity('Error unarchiving project: ' + error, 'error');
    }
}

async function archiveProject(projectId) {
    if (!confirm('Archive this project?')) return;
    
    try {
        const res = await fetch(`/api/project/${projectId}/archive`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ archive: true })
        });
        const data = await safeJsonParse(res);
        
        if (data.status === 'archived') {
            addActivity('Project archived');
            loadSidebarProjects();
            loadWorkflow();
        } else {
            addActivity('Error archiving project: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        addActivity('Error archiving project: ' + error, 'error');
    }
}

async function deleteProject(projectId) {
    if (!confirm('Permanently delete this project? This cannot be undone.')) return;
    
    try {
        const res = await fetch(`/api/project/${projectId}`, {
            method: 'DELETE'
        });
        const data = await safeJsonParse(res);
        
        if (data.status === 'deleted' || !data.error) {
            addActivity('Project deleted');
            loadArchive();
            loadSidebarProjects();
        } else {
            addActivity('Error deleting project: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        addActivity('Error deleting project: ' + error, 'error');
    }
}

// Make functions global for onclick handlers
window.runTask = runTask;
window.viewTaskOutput = viewTaskOutput;
window.renameProject = renameProject;
window.unarchiveProject = unarchiveProject;
window.archiveProject = archiveProject;
window.deleteProject = deleteProject;
window.quickRename = quickRename;

// Onboarding Wizard
let onboardingStep = 0;
const onboardingSteps = [
    {
        title: "Welcome to Live Model Discovery",
        content: `
            <p>This wizard will guide you through setting up live model discovery for Vetinari.</p>
            <p>You'll be able to search HuggingFace, Reddit, GitHub, and Papers With Code for the best models.</p>
        `
    },
    {
        title: "Step 1: Add API Tokens",
        content: `
            <p>To search external sources, you need to add API tokens:</p>
            <ul>
                <li><strong>HuggingFace:</strong> Get a token from huggingface.co/settings/tokens</li>
                <li><strong>Reddit:</strong> Create a script app at reddit.com/prefs/apps</li>
                <li><strong>GitHub:</strong> Generate a token at github.com/settings/tokens</li>
            </ul>
            <p>Go to <strong>Settings > Admin Credentials</strong> to add your tokens.</p>
        `
    },
    {
        title: "Step 2: Configure Rotation",
        content: `
            <p>Set a rotation period for each token (default: 30 days).</p>
            <p>Vetinari will remind you when tokens need rotation to keep your discovery working.</p>
        `
    },
    {
        title: "Step 3: Test Model Search",
        content: `
            <p>Once tokens are configured:</p>
            <ul>
                <li>Click any task in the Tasks panel</li>
                <li>See model recommendations with scores</li>
                <li>Hover over candidates to see why they were ranked</li>
                <li>Use "Apply Override" to pin a model to a task</li>
            </ul>
        `
    },
    {
        title: "You're Ready!",
        content: `
            <p>That's it! Live model discovery is now available.</p>
            <p>Remember:</p>
            <ul>
                <li>Monthly refresh keeps recommendations updated</li>
                <li>You can disable per-project in project.yaml</li>
                <li>Check docs/setup/onboarding.md for more details</li>
            </ul>
        `
    }
];

function showOnboarding() {
    const modal = document.getElementById('onboardingModal');
    if (!modal) return;
    
    onboardingStep = 0;
    renderOnboardingStep();
    modal.style.display = 'flex';
}

function closeOnboarding() {
    const modal = document.getElementById('onboardingModal');
    if (modal) {
        modal.style.display = 'none';
    }
    localStorage.setItem('onboardingComplete', 'true');
}

function renderOnboardingStep() {
    const container = document.getElementById('onboardingSteps');
    if (!container) return;
    
    const step = onboardingSteps[onboardingStep];
    const total = onboardingSteps.length;
    
    container.innerHTML = `
        <div class="onboarding-progress">
            ${onboardingSteps.map((_, i) => `<span class="onboarding-dot ${i === onboardingStep ? 'active' : ''}"></span>`).join('')}
        </div>
        <div class="onboarding-step active">
            <h3>${step.title}</h3>
            ${step.content}
        </div>
    `;
    
    document.getElementById('onboardingPrev').style.display = onboardingStep > 0 ? 'inline-block' : 'none';
    document.getElementById('onboardingNext').textContent = onboardingStep === total - 1 ? 'Finish' : 'Next';
}

function onboardingNext() {
    if (onboardingStep < onboardingSteps.length - 1) {
        onboardingStep++;
        renderOnboardingStep();
    } else {
        closeOnboarding();
    }
}

function onboardingPrev() {
    if (onboardingStep > 0) {
        onboardingStep--;
        renderOnboardingStep();
    }
}

// Check if onboarding should be shown
function checkOnboarding() {
    const isComplete = localStorage.getItem('onboardingComplete');
    if (!isComplete) {
        fetch('/api/admin/permissions')
            .then(res => res.json())
            .then(data => {
                if (data.admin) {
                    setTimeout(showOnboarding, 1000);
                }
            })
            .catch(() => {});
    }
}

// Initialize onboarding check
document.addEventListener('DOMContentLoaded', checkOnboarding);
window.closeOnboarding = closeOnboarding;
window.onboardingNext = onboardingNext;
window.onboardingPrev = onboardingPrev;
window.showOnboarding = showOnboarding;
