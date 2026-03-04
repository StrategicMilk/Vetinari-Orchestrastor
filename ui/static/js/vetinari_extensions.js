/**
 * Vetinari UI Extensions
 * =======================
 * New and improved functionality that extends/overrides the base app.js.
 * Loaded AFTER app.js so it can override globals.
 *
 * Features added here:
 *   - Project Intake Form submission
 *   - Improved chat message rendering (markdown + streaming)
 *   - Todo task list updates
 *   - Goal verification UI
 *   - Rules configuration
 *   - Image generation UI
 *   - Training UI
 *   - Hardware detection
 *   - Replace window.prompt/alert/confirm with custom modals
 */

'use strict';

// ──────────────────────────────────────────────────────────────────────────────
// Utility: Custom modal dialogs (replaces window.alert/prompt/confirm)
// ──────────────────────────────────────────────────────────────────────────────

const VModal = {
    _container: null,

    _getContainer() {
        if (!this._container) {
            this._container = document.createElement('div');
            this._container.id = 'vModalContainer';
            document.body.appendChild(this._container);
        }
        return this._container;
    },

    /**
     * Show a confirmation dialog. Returns a Promise<boolean>.
     */
    confirm(message, title = 'Confirm') {
        return new Promise((resolve) => {
            const id = 'vmc_' + Date.now();
            const html = `
<div class="modal-backdrop" id="${id}_backdrop" onclick="VModal._close('${id}', false)">
    <div class="modal-content" onclick="event.stopPropagation()" style="max-width:420px;">
        <div class="modal-header">
            <h3>${escapeHtml(title)}</h3>
        </div>
        <div class="modal-body">
            <p>${escapeHtml(message)}</p>
        </div>
        <div class="modal-footer">
            <button class="btn btn-secondary" onclick="VModal._close('${id}', false)">Cancel</button>
            <button class="btn btn-danger" onclick="VModal._close('${id}', true)">Confirm</button>
        </div>
    </div>
</div>`;
            this._getContainer().insertAdjacentHTML('beforeend', html);
            this._resolvers = this._resolvers || {};
            this._resolvers[id] = resolve;
        });
    },

    /**
     * Show a prompt dialog. Returns a Promise<string|null>.
     */
    prompt(message, defaultValue = '', title = 'Input') {
        return new Promise((resolve) => {
            const id = 'vmp_' + Date.now();
            const html = `
<div class="modal-backdrop" id="${id}_backdrop">
    <div class="modal-content" onclick="event.stopPropagation()" style="max-width:460px;">
        <div class="modal-header">
            <h3>${escapeHtml(title)}</h3>
        </div>
        <div class="modal-body">
            <p>${escapeHtml(message)}</p>
            <input type="text" class="input" id="${id}_input" value="${escapeHtml(defaultValue)}"
                style="width:100%;margin-top:0.5rem;"
                onkeydown="if(event.key==='Enter')VModal._closePrompt('${id}');if(event.key==='Escape')VModal._close('${id}',null)">
        </div>
        <div class="modal-footer">
            <button class="btn btn-secondary" onclick="VModal._close('${id}', null)">Cancel</button>
            <button class="btn btn-primary" onclick="VModal._closePrompt('${id}')">OK</button>
        </div>
    </div>
</div>`;
            this._getContainer().insertAdjacentHTML('beforeend', html);
            this._resolvers = this._resolvers || {};
            this._resolvers[id] = resolve;
            setTimeout(() => document.getElementById(`${id}_input`)?.focus(), 50);
        });
    },

    /** Close a modal by id with a value. */
    _close(id, value) {
        document.getElementById(`${id}_backdrop`)?.remove();
        if (this._resolvers && this._resolvers[id]) {
            this._resolvers[id](value);
            delete this._resolvers[id];
        }
    },

    _closePrompt(id) {
        const val = document.getElementById(`${id}_input`)?.value ?? null;
        this._close(id, val);
    }
};

// Make VModal global
window.VModal = VModal;

// ──────────────────────────────────────────────────────────────────────────────
// Improved message renderer (uses marked.js if available)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Override the existing formatMessageContent with a markdown-aware version.
 */
window.formatMessageContent = function(content) {
    if (!content) return '';
    if (typeof window.renderMarkdown === 'function') {
        return window.renderMarkdown(content);
    }
    // Fallback: basic HTML escaping + newlines
    return escapeHtml(content).replace(/\n/g, '<br>');
};

/**
 * Append a message to the chat with proper markdown rendering.
 */
function appendChatMessage(role, content, extra = {}) {
    const messagesEl = document.getElementById('chatMessages');
    if (!messagesEl) return;

    // Hide empty state
    const emptyEl = document.getElementById('chatEmpty');
    if (emptyEl) emptyEl.style.display = 'none';

    const div = document.createElement('div');
    div.className = `chat-message ${role}`;
    if (extra.id) div.id = extra.id;

    const avatarIcon = role === 'user' ? 'fa-user' : 'fa-robot';
    const contentClass = extra.error ? 'error' : (extra.warning ? 'warning' : '');

    div.innerHTML = `
        <div class="chat-message-avatar"><i class="fas ${avatarIcon}"></i></div>
        <div class="chat-message-body">
            <div class="chat-message-content markdown-body ${contentClass}">${formatMessageContent(content)}</div>
            ${extra.timestamp ? `<div class="chat-message-meta">${extra.timestamp}</div>` : ''}
        </div>
    `;

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Re-run syntax highlighting on new code blocks
    if (typeof hljs !== 'undefined') {
        div.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    }
    // Re-run mermaid on diagram blocks
    if (typeof mermaid !== 'undefined') {
        div.querySelectorAll('.language-mermaid').forEach(el => {
            mermaid.run({ nodes: [el] });
        });
    }

    return div;
}
window.appendChatMessage = appendChatMessage;

/**
 * Show typing indicator in chat.
 */
function showTypingIndicator() {
    const messagesEl = document.getElementById('chatMessages');
    if (!messagesEl) return null;
    const emptyEl = document.getElementById('chatEmpty');
    if (emptyEl) emptyEl.style.display = 'none';
    const div = document.createElement('div');
    div.className = 'chat-message assistant';
    div.id = 'typingIndicator';
    div.innerHTML = `
        <div class="chat-message-avatar"><i class="fas fa-robot"></i></div>
        <div class="chat-message-body">
            <div class="chat-message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>`;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
}

function hideTypingIndicator() {
    document.getElementById('typingIndicator')?.remove();
}

// ──────────────────────────────────────────────────────────────────────────────
// Task Todo List (OpenCode-inspired)
// ──────────────────────────────────────────────────────────────────────────────

const STATUS_ICONS = {
    pending:     '<i class="fas fa-circle task-status-icon pending" title="Pending"></i>',
    ready:       '<i class="fas fa-circle task-status-icon pending" title="Ready"></i>',
    in_progress: '<i class="fas fa-circle-notch fa-spin task-status-icon in_progress" title="Running"></i>',
    running:     '<i class="fas fa-circle-notch fa-spin task-status-icon in_progress" title="Running"></i>',
    completed:   '<i class="fas fa-check-circle task-status-icon completed" title="Done"></i>',
    failed:      '<i class="fas fa-times-circle task-status-icon failed" title="Failed"></i>',
    cancelled:   '<i class="fas fa-ban task-status-icon cancelled" title="Cancelled"></i>',
    blocked:     '<i class="fas fa-lock task-status-icon cancelled" title="Blocked"></i>',
};

/**
 * Render the task todo list - improved version that matches the new CSS.
 */
window.renderTasks = function(tasks) {
    const container = document.getElementById('tasksList');
    const chatTasksEl = document.getElementById('chatTasks');
    const badge = document.getElementById('tasksCountBadge');

    if (!container) return;

    if (!tasks || tasks.length === 0) {
        if (chatTasksEl) chatTasksEl.style.display = 'none';
        return;
    }

    if (chatTasksEl) chatTasksEl.style.display = 'block';
    if (badge) badge.textContent = tasks.length;

    container.innerHTML = tasks.map(task => {
        const status = (task.status || 'pending').toLowerCase().replace(' ', '_');
        const icon = STATUS_ICONS[status] || STATUS_ICONS.pending;
        const agent = task.assigned_agent || task.agent_type || '';
        const model = task.assigned_model || task.assigned_model_id || '';
        const desc = task.description || task.name || 'Task';
        const taskId = task.id || task.task_id || '';

        return `
<div class="task-item ${status}" role="listitem" id="task-item-${escapeHtml(taskId)}">
    <span class="task-status-icon">${icon}</span>
    <div class="task-info">
        <div class="task-description" title="${escapeHtml(desc)}">${escapeHtml(desc)}</div>
        <div class="task-meta">
            ${agent ? `<span class="task-agent-badge">${escapeHtml(agent)}</span>` : ''}
            ${model ? `<span class="task-agent-badge" style="color:var(--secondary)">${escapeHtml(model)}</span>` : ''}
        </div>
    </div>
    ${task.output || task.error ? `
    <button class="task-expand" onclick="toggleTaskDetails('${escapeHtml(taskId)}')" title="Show details">
        <i class="fas fa-chevron-down"></i>
    </button>` : ''}
</div>
${task.output || task.error ? `
<div class="task-details" id="task-details-${escapeHtml(taskId)}">
    ${task.error ? `<div style="color:var(--danger)"><strong>Error:</strong> ${escapeHtml(task.error)}</div>` : ''}
    ${task.output ? `<div>${escapeHtml(String(task.output).slice(0, 300))}...</div>` : ''}
</div>` : ''}`;
    }).join('');
};

function toggleTaskDetails(taskId) {
    const el = document.getElementById(`task-details-${taskId}`);
    if (el) el.classList.toggle('open');
}
window.toggleTaskDetails = toggleTaskDetails;

/**
 * Update a single task's status in the todo list.
 */
function updateTaskStatus(taskId, status, extra = {}) {
    const item = document.getElementById(`task-item-${taskId}`);
    if (!item) return;

    // Update class
    item.className = `task-item ${status.toLowerCase()}`;

    // Update icon
    const iconEl = item.querySelector('.task-status-icon');
    if (iconEl) {
        const normalizedStatus = status.toLowerCase().replace(' ', '_');
        iconEl.outerHTML = STATUS_ICONS[normalizedStatus] || STATUS_ICONS.pending;
    }

    // Update description strikethrough
    const desc = item.querySelector('.task-description');
    if (desc && status === 'completed') {
        desc.style.textDecoration = 'line-through';
        desc.style.color = 'var(--text-muted)';
    }
}
window.updateTaskStatus = updateTaskStatus;

// ──────────────────────────────────────────────────────────────────────────────
// Project Intake Form
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Collect the intake form data into a structured object.
 */
function collectIntakeFormData() {
    const name = document.getElementById('intakeProjectName')?.value.trim() || '';
    const goal = document.getElementById('intakeGoal')?.value.trim() || '';
    const details = document.getElementById('intakeDetails')?.value.trim() || '';
    const featuresRaw = document.getElementById('intakeFeatures')?.value || '';
    const avoidRaw = document.getElementById('intakeAvoid')?.value || '';
    const tech = document.getElementById('intakeTech')?.value.trim() || '';
    const priority = document.getElementById('intakePriority')?.value || 'quality';
    const model = document.getElementById('intakeModelSelect')?.value || '';

    // Parse features/avoid as lists
    const features = featuresRaw.split('\n')
        .map(l => l.replace(/^[-*•]\s*/, '').trim())
        .filter(Boolean);
    const avoid = avoidRaw.split('\n')
        .map(l => l.replace(/^[-*•]\s*/, '').trim())
        .filter(Boolean);

    // Collect checkboxes
    const platforms = Array.from(
        document.querySelectorAll('#intakePlatform input:checked')
    ).map(cb => cb.value);
    const expected = Array.from(
        document.querySelectorAll('#intakeExpected input:checked')
    ).map(cb => cb.value);

    // Build combined goal string
    let fullGoal = goal;
    if (details) fullGoal += '\n\nDetails: ' + details;
    if (tech) fullGoal += '\nTechnologies: ' + tech;
    if (platforms.length) fullGoal += '\nTarget platform: ' + platforms.join(', ');
    if (priority) fullGoal += '\nPriority: ' + priority;

    return {
        project_name: name || goal.slice(0, 50),
        goal: fullGoal,
        raw_goal: goal,
        details,
        required_features: features,
        things_to_avoid: avoid,
        tech_stack: tech,
        platforms,
        priority,
        expected_outputs: expected,
        model,
        system_prompt: document.getElementById('systemPromptInput')?.value || '',
        project_rules: document.getElementById('projectRulesInput')?.value || '',
    };
}

/**
 * Clear the intake form.
 */
function clearIntakeForm() {
    ['intakeProjectName', 'intakeGoal', 'intakeDetails', 'intakeFeatures',
     'intakeAvoid', 'intakeTech'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    document.querySelectorAll('#intakePlatform input, #intakeExpected input')
        .forEach(cb => cb.checked = false);
    // Re-check defaults
    ['code', 'tests'].forEach(val => {
        const cb = document.querySelector(`#intakeExpected input[value="${val}"]`);
        if (cb) cb.checked = true;
    });
}
window.clearIntakeForm = clearIntakeForm;

/**
 * Submit the intake form and create a new project.
 */
async function submitIntakeForm() {
    const submitBtn = document.getElementById('intakeSubmitBtn');
    const formData = collectIntakeFormData();

    if (!formData.raw_goal) {
        showStatusBanner('Please enter a goal for your project.', 'warning');
        document.getElementById('intakeGoal')?.focus();
        return;
    }

    // Show chat area, hide form
    const form = document.getElementById('projectIntakeForm');
    const chatInputArea = document.getElementById('chatInputArea');
    const messagesEl = document.getElementById('chatMessages');
    const projectTitle = document.getElementById('chatProjectTitle');

    if (form) form.style.display = 'none';
    if (chatInputArea) chatInputArea.style.display = 'flex';
    if (projectTitle) projectTitle.textContent = formData.project_name;

    // Show user's goal as first message
    appendChatMessage('user', formData.raw_goal);
    const typingDiv = showTypingIndicator();

    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating...';
    }

    // Show progress section
    const progressSection = document.getElementById('progressSection');
    if (progressSection) progressSection.style.display = 'block';
    updateProgressBar(0);
    const workingOnText = document.getElementById('workingOnText');
    if (workingOnText) workingOnText.innerHTML = '<i class="fas fa-cog fa-spin"></i> Analyzing goal and planning tasks...';

    try {
        const res = await fetch('/api/new-project', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                goal: formData.goal,
                project_name: formData.project_name,
                model: formData.model,
                system_prompt: formData.system_prompt,
                project_rules: formData.project_rules,
                required_features: formData.required_features,
                things_to_avoid: formData.things_to_avoid,
                expected_outputs: formData.expected_outputs,
                tech_stack: formData.tech_stack,
                platforms: formData.platforms,
                priority: formData.priority,
                auto_run: true,
            })
        });
        const data = await safeJsonParse(res);

        hideTypingIndicator();

        if (!data) {
            appendChatMessage('assistant', 'No response from server. Please check your LM Studio connection.', { error: true });
            return;
        }

        if (data.error) {
            appendChatMessage('assistant', `Error: ${data.error}`, { error: true });
            addActivity('Error creating project: ' + data.error, 'error');
            return;
        }

        if (data.needs_context) {
            appendChatMessage('assistant', data.follow_up_question || 'Could you provide more details?');
            // Re-show intake form with question highlighted
            if (form) form.style.display = 'flex';
            if (chatInputArea) chatInputArea.style.display = 'none';
            showStatusBanner(data.follow_up_question || 'More details needed', 'info');
            return;
        }

        if (data.project_id) {
            window.currentProjectId = data.project_id;

            // Render conversation
            if (data.conversation && data.conversation.length) {
                renderConversation(data.conversation);
            } else if (data.initial_response) {
                appendChatMessage('assistant', data.initial_response);
            } else {
                appendChatMessage('assistant', `Project created! Planning ${data.tasks?.length || 0} tasks...`);
            }

            // Render tasks todo list
            if (data.tasks && data.tasks.length) {
                renderTasks(data.tasks);
            }

            // Save required_features to project for later verification
            window._currentProjectSpec = {
                goal: formData.raw_goal,
                required_features: formData.required_features,
                things_to_avoid: formData.things_to_avoid,
                expected_outputs: formData.expected_outputs,
            };

            // Subscribe to SSE stream for real-time updates
            subscribeToProjectStream(data.project_id);

            showStatusBanner(`Project created: ${formData.project_name}`, 'success');
            addActivity(`New project: ${formData.project_name} (${data.tasks?.length || 0} tasks)`);

            // Refresh sidebar
            loadSidebarProjects();
        }

    } catch (err) {
        hideTypingIndicator();
        console.error('Intake form submit error:', err);
        appendChatMessage('assistant', `Failed to create project: ${err.message}`, { error: true });
        addActivity('Error: ' + err.message, 'error');
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-rocket"></i> Launch Project';
        }
    }
}
window.submitIntakeForm = submitIntakeForm;

// Show intake form when "New Project" is clicked
function showNewProjectForm() {
    window.currentProjectId = null;

    const form = document.getElementById('projectIntakeForm');
    const chatInputArea = document.getElementById('chatInputArea');
    const messagesEl = document.getElementById('chatMessages');
    const chatTasks = document.getElementById('chatTasks');
    const progressSection = document.getElementById('progressSection');
    const finalDelivery = document.getElementById('finalDeliveryPanel');
    const modelRanking = document.getElementById('modelRankingPanel');
    const chatProjectTitle = document.getElementById('chatProjectTitle');

    if (form) form.style.display = 'flex';
    if (chatInputArea) chatInputArea.style.display = 'none';
    if (chatTasks) chatTasks.style.display = 'none';
    if (progressSection) progressSection.style.display = 'none';
    if (finalDelivery) finalDelivery.style.display = 'none';
    if (modelRanking) modelRanking.style.display = 'none';
    if (chatProjectTitle) chatProjectTitle.textContent = 'New Project';

    if (messagesEl) {
        messagesEl.innerHTML = '<div class="chat-empty" id="chatEmpty"><i class="fas fa-comments"></i><p>Select a project or create a new one</p></div>';
    }

    // Switch to prompt view
    switchView('prompt');
}
window.showNewProjectForm = showNewProjectForm;

// ──────────────────────────────────────────────────────────────────────────────
// SSE enhanced handler — updates task list in real-time
// ──────────────────────────────────────────────────────────────────────────────

// Override the existing SSE handler to update the todo list
const _originalSubscribe = window.subscribeToProjectStream;
window.subscribeToProjectStream = function(projectId) {
    // Call original subscription if it exists
    if (typeof _originalSubscribe === 'function') {
        _originalSubscribe(projectId);
    }

    // Also set up our enhanced task-status updater
    if (!projectId) return;

    // Listen for SSE events already being processed by app.js
    // and additionally update our new task list
    document.addEventListener(`sse:task_started:${projectId}`, (e) => {
        updateTaskStatus(e.detail.task_id, 'in_progress');
        const workingOnText = document.getElementById('workingOnText');
        if (workingOnText) {
            workingOnText.innerHTML = `<i class="fas fa-cog fa-spin"></i> Working on: ${escapeHtml(e.detail.description || e.detail.task_id)}`;
        }
    });

    document.addEventListener(`sse:task_completed:${projectId}`, (e) => {
        updateTaskStatus(e.detail.task_id, 'completed');
    });

    document.addEventListener(`sse:task_failed:${projectId}`, (e) => {
        updateTaskStatus(e.detail.task_id, 'failed');
    });

    document.addEventListener(`sse:task_cancelled:${projectId}`, (e) => {
        updateTaskStatus(e.detail.task_id, 'cancelled');
    });
};

// ──────────────────────────────────────────────────────────────────────────────
// Goal Verification UI
// ──────────────────────────────────────────────────────────────────────────────

async function runGoalVerification() {
    if (!window.currentProjectId) return;

    const spec = window._currentProjectSpec || {};
    const finalContent = document.getElementById('finalDeliveryContent')?.innerText || '';

    showStatusBanner('Verifying deliverable against goal...', 'info');

    try {
        const res = await fetch(`/api/project/${currentProjectId}/verify-goal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                goal: spec.goal || '',
                final_output: finalContent,
                required_features: spec.required_features || [],
                things_to_avoid: spec.things_to_avoid || [],
                expected_outputs: spec.expected_outputs || [],
            })
        });
        const data = await safeJsonParse(res);

        if (!data || data.error) {
            showStatusBanner('Verification failed: ' + (data?.error || 'Unknown error'), 'error');
            return;
        }

        renderGoalVerification(data.report);

        // If corrective tasks needed, offer to re-execute
        if (!data.report.fully_compliant && data.corrective_tasks?.length) {
            appendChatMessage('assistant',
                `**Verification found ${data.corrective_tasks.length} gaps:**\n` +
                data.corrective_tasks.map(t => `- ${t.description}`).join('\n') +
                '\n\nShould I fix these issues? (yes/no)',
                { timestamp: new Date().toLocaleTimeString() }
            );
        } else if (data.report.fully_compliant) {
            showStatusBanner('Deliverable fully meets the project goal!', 'success');
        }
    } catch (err) {
        console.error('Goal verification error:', err);
        showStatusBanner('Verification error: ' + err.message, 'error');
    }
}
window.runGoalVerification = runGoalVerification;

function renderGoalVerification(report) {
    const verificationEl = document.getElementById('goalVerification');
    const matrixEl = document.getElementById('verificationMatrix');
    const finalPanel = document.getElementById('finalDeliveryPanel');

    if (!verificationEl || !matrixEl) return;

    verificationEl.style.display = 'block';
    if (finalPanel) finalPanel.style.display = 'block';

    const scorePercent = Math.round((report.compliance_score || 0) * 100);
    const scoreColor = scorePercent >= 80 ? 'var(--success)' : scorePercent >= 60 ? 'var(--warning)' : 'var(--danger)';

    matrixEl.innerHTML = `
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem;padding:0.5rem;background:var(--surface-bg);border-radius:var(--radius-md);">
            <span style="font-size:1.5rem;font-weight:700;color:${scoreColor}">${scorePercent}%</span>
            <span style="color:var(--text-secondary);font-size:0.875rem;">Compliance Score</span>
            <span class="badge ${report.fully_compliant ? 'badge-success' : 'badge-warning'}" style="margin-left:auto;">
                ${report.fully_compliant ? '✓ Compliant' : '⚠ Gaps Found'}
            </span>
        </div>
        ${(report.features || []).map(f => `
        <div class="verification-row">
            <span class="verification-status">${f.implemented ? '✅' : '❌'}</span>
            <span class="verification-feature">${escapeHtml(f.feature)}</span>
            <span class="verification-evidence">${escapeHtml(f.evidence || '')}</span>
        </div>`).join('')}
        ${!report.security_passed ? `
        <div class="verification-row" style="border-color:var(--danger);">
            <span class="verification-status">🔒</span>
            <span class="verification-feature">Security: ${report.security_findings?.length || 0} findings</span>
            <span class="verification-evidence" style="color:var(--danger);">Failed security audit</span>
        </div>` : ''}
        ${!report.tests_present ? `
        <div class="verification-row">
            <span class="verification-status">⚠</span>
            <span class="verification-feature">Tests</span>
            <span class="verification-evidence">No tests detected</span>
        </div>` : ''}
    `;
}

function approveDeliverable() {
    if (!window.currentProjectId) return;
    fetch(`/api/project/${currentProjectId}/approve`, { method: 'POST' })
        .then(r => r.json())
        .then(() => {
            showStatusBanner('Deliverable approved!', 'success');
            appendChatMessage('assistant', '✅ Deliverable approved and saved. Great work!');
        })
        .catch(err => showStatusBanner('Approval failed: ' + err.message, 'error'));
}
window.approveDeliverable = approveDeliverable;

function requestChanges() {
    VModal.prompt('What changes would you like to make?', '', 'Request Changes')
        .then(changes => {
            if (!changes) return;
            const input = document.getElementById('chatInput');
            if (input) {
                input.value = changes;
                // Trigger send
                document.getElementById('sendMessageBtn')?.click();
            }
        });
}
window.requestChanges = requestChanges;

function copyDeliverable() {
    const content = document.getElementById('finalDeliveryContent')?.innerText || '';
    navigator.clipboard.writeText(content).then(() => {
        showStatusBanner('Deliverable copied to clipboard', 'success');
    });
}
window.copyDeliverable = copyDeliverable;

// ──────────────────────────────────────────────────────────────────────────────
// Rules Configuration
// ──────────────────────────────────────────────────────────────────────────────

async function loadGlobalRules() {
    try {
        const res = await fetch('/api/rules/global');
        const data = await res.json();
        const rules = (data.rules || []).join('\n');
        const input = document.getElementById('projectRulesInput');
        if (input && rules) input.value = rules;
    } catch (err) {
        console.error('Failed to load global rules:', err);
    }
}
window.loadGlobalRules = loadGlobalRules;

async function saveProjectRules() {
    if (!window.currentProjectId) {
        showStatusBanner('Select a project first', 'warning');
        return;
    }
    const rulesText = document.getElementById('projectRulesInput')?.value || '';
    const rules = rulesText.split('\n').map(r => r.trim()).filter(Boolean);
    try {
        await fetch(`/api/rules/project/${currentProjectId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rules }),
        });
        showStatusBanner('Project rules saved', 'success');
    } catch (err) {
        showStatusBanner('Failed to save rules: ' + err.message, 'error');
    }
}
window.saveProjectRules = saveProjectRules;

async function saveGlobalRules() {
    const rulesText = document.getElementById('globalRulesInput')?.value || '';
    const rules = rulesText.split('\n').map(r => r.trim()).filter(Boolean);
    try {
        await fetch('/api/rules/global', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rules }),
        });
        showStatusBanner('Global rules saved', 'success');
    } catch (err) {
        showStatusBanner('Failed to save global rules: ' + err.message, 'error');
    }
}
window.saveGlobalRules = saveGlobalRules;

async function saveGlobalSystemPrompt() {
    const prompt = document.getElementById('globalSystemPromptInput')?.value || '';
    try {
        await fetch('/api/rules/global-prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt }),
        });
        showStatusBanner('Global system prompt saved', 'success');
    } catch (err) {
        showStatusBanner('Failed to save: ' + err.message, 'error');
    }
}
window.saveGlobalSystemPrompt = saveGlobalSystemPrompt;

async function loadModelRules() {
    const modelId = document.getElementById('modelRulesModelSelect')?.value;
    if (!modelId) return;
    try {
        const res = await fetch(`/api/rules/model/${encodeURIComponent(modelId)}`);
        const data = await res.json();
        const input = document.getElementById('modelRulesInput');
        if (input) input.value = (data.rules || []).join('\n');
    } catch (err) {
        showStatusBanner('Failed to load model rules: ' + err.message, 'error');
    }
}
window.loadModelRules = loadModelRules;

async function saveModelRules() {
    const modelId = document.getElementById('modelRulesModelSelect')?.value;
    if (!modelId) { showStatusBanner('Select a model first', 'warning'); return; }
    const rulesText = document.getElementById('modelRulesInput')?.value || '';
    const rules = rulesText.split('\n').map(r => r.trim()).filter(Boolean);
    try {
        await fetch(`/api/rules/model/${encodeURIComponent(modelId)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rules }),
        });
        showStatusBanner('Model rules saved', 'success');
    } catch (err) {
        showStatusBanner('Failed to save model rules: ' + err.message, 'error');
    }
}
window.saveModelRules = saveModelRules;

// ──────────────────────────────────────────────────────────────────────────────
// Hardware Detection
// ──────────────────────────────────────────────────────────────────────────────

async function detectHardware() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        const gpu = data.gpu_name || data.hardware?.gpu || 'Unknown GPU';
        const vram = data.vram_total_gb ? `${data.vram_total_gb} GB` : (data.hardware?.vram || '--');
        const ram = data.ram_gb ? `${data.ram_gb} GB` : (data.hardware?.ram || '--');

        document.getElementById('hwGpu').textContent = gpu;
        document.getElementById('hwVram').textContent = vram;
        document.getElementById('hwRam').textContent = ram;
    } catch (err) {
        document.getElementById('hwGpu').textContent = 'Not detected';
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Training UI
// ──────────────────────────────────────────────────────────────────────────────

async function loadTrainingStats() {
    try {
        const res = await fetch('/api/training/stats');
        const data = await res.json();
        const el = document.getElementById('trainingStats');
        if (!el) return;
        el.innerHTML = `
            <div class="training-stat-item">
                <span class="training-stat-label">Total Records</span>
                <span class="training-stat-value">${data.total_records || 0}</span>
            </div>
            <div class="training-stat-item">
                <span class="training-stat-label">Avg Quality</span>
                <span class="training-stat-value">${((data.avg_quality || 0) * 100).toFixed(1)}%</span>
            </div>
            <div class="training-stat-item">
                <span class="training-stat-label">Models</span>
                <span class="training-stat-value">${data.unique_models || 0}</span>
            </div>
            <div class="training-stat-item">
                <span class="training-stat-label">Task Types</span>
                <span class="training-stat-value">${data.unique_task_types || 0}</span>
            </div>
        `;
    } catch (err) {
        document.getElementById('trainingStats')?.innerHTML = '<span style="color:var(--text-muted)">No training data yet</span>';
    }
}

async function exportTrainingData() {
    try {
        const format = 'sft';
        showStatusBanner('Exporting training data...', 'info');
        const res = await fetch('/api/training/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ format })
        });
        const data = await res.json();
        showStatusBanner(`Exported ${data.count} training records`, 'success');
    } catch (err) {
        showStatusBanner('Export failed: ' + err.message, 'error');
    }
}
window.exportTrainingData = exportTrainingData;

async function startTraining() {
    const tier = document.getElementById('trainingTierSelect')?.value || 'general';
    const modelId = document.getElementById('trainingModelSelect')?.value || '';
    const confirmed = await VModal.confirm(
        `Start a ${tier} training run${modelId ? ` for ${modelId}` : ''}? This may take a while.`,
        'Start Training'
    );
    if (!confirmed) return;
    try {
        const res = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tier, model_id: modelId })
        });
        const data = await res.json();
        showStatusBanner(data.message || 'Training started', 'success');
    } catch (err) {
        showStatusBanner('Failed to start training: ' + err.message, 'error');
    }
}
window.startTraining = startTraining;

// Show/hide individual model select based on tier
document.getElementById('trainingTierSelect')?.addEventListener('change', function() {
    const group = document.getElementById('trainingModelSelectGroup');
    if (group) group.style.display = this.value === 'individual' ? 'block' : 'none';
});

// ──────────────────────────────────────────────────────────────────────────────
// Image Generation UI
// ──────────────────────────────────────────────────────────────────────────────

async function testSdConnection() {
    const host = document.getElementById('sdHostInput')?.value.trim() || 'http://localhost:7860';
    const statusEl = document.getElementById('sdStatus');
    if (statusEl) statusEl.textContent = 'Testing...';

    try {
        const res = await fetch('/api/sd-status');
        const data = await res.json();
        if (statusEl) {
            statusEl.className = `sd-status ${data.status === 'connected' ? 'ok' : 'error'}`;
            statusEl.textContent = data.status === 'connected'
                ? `✓ Connected to ${data.host}`
                : `✗ ${data.error || 'Disconnected'}`;
        }
    } catch (err) {
        if (statusEl) {
            statusEl.className = 'sd-status error';
            statusEl.textContent = '✗ Connection failed: ' + err.message;
        }
    }
}
window.testSdConnection = testSdConnection;

function saveSdSettings() {
    // Store in localStorage for now (proper backend saving TODO)
    const host = document.getElementById('sdHostInput')?.value.trim() || '';
    if (host) localStorage.setItem('vetinari_sd_host', host);
    showStatusBanner('Image generation settings saved', 'success');
}
window.saveSdSettings = saveSdSettings;

// ──────────────────────────────────────────────────────────────────────────────
// Password visibility toggle
// ──────────────────────────────────────────────────────────────────────────────

function togglePasswordVis(inputId, btn) {
    const input = document.getElementById(inputId);
    if (!input) return;
    if (input.type === 'password') {
        input.type = 'text';
        btn.innerHTML = '<i class="fas fa-eye-slash"></i>';
    } else {
        input.type = 'password';
        btn.innerHTML = '<i class="fas fa-eye"></i>';
    }
}
window.togglePasswordVis = togglePasswordVis;

// ──────────────────────────────────────────────────────────────────────────────
// Rules panel toggles
// ──────────────────────────────────────────────────────────────────────────────

document.getElementById('toggleRulesBtn')?.addEventListener('click', () => {
    const panel = document.getElementById('rulesPanel');
    if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
});

// ──────────────────────────────────────────────────────────────────────────────
// Wire up the intake form submit button and "New Project" buttons
// ──────────────────────────────────────────────────────────────────────────────

document.getElementById('intakeSubmitBtn')?.addEventListener('click', submitIntakeForm);

// Override existing newProjectBtn to show the intake form
const _existingNewProjectBtn = document.getElementById('newProjectBtn');
if (_existingNewProjectBtn) {
    _existingNewProjectBtn.removeEventListener('click', _existingNewProjectBtn.__handler);
    _existingNewProjectBtn.addEventListener('click', showNewProjectForm);
}

// Keyboard shortcut: Ctrl+Enter to submit intake form
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const intakeForm = document.getElementById('projectIntakeForm');
        if (intakeForm && intakeForm.style.display !== 'none') {
            submitIntakeForm();
            return;
        }
        // If chat input is focused
        const chatInput = document.getElementById('chatInput');
        if (document.activeElement === chatInput) {
            window.sendChatMessage?.();
        }
    }
});

// ──────────────────────────────────────────────────────────────────────────────
// Settings page: load rules and training data on view switch
// ──────────────────────────────────────────────────────────────────────────────

const _originalSwitchView = window.switchView;
window.switchView = function(viewId) {
    if (typeof _originalSwitchView === 'function') {
        _originalSwitchView(viewId);
    }
    if (viewId === 'settings') {
        loadSettingsRules();
        loadTrainingStats();
        loadSdSettings();
    }
    if (viewId === 'prompt') {
        const form = document.getElementById('projectIntakeForm');
        if (form && !window.currentProjectId) {
            form.style.display = 'flex';
            detectHardware();
        }
    }
};

async function loadSettingsRules() {
    try {
        const res = await fetch('/api/rules');
        const data = await res.json();

        const globalRulesInput = document.getElementById('globalRulesInput');
        if (globalRulesInput) globalRulesInput.value = (data.global || []).join('\n');

        const globalPromptInput = document.getElementById('globalSystemPromptInput');
        if (globalPromptInput) globalPromptInput.value = data.global_system_prompt || '';
    } catch (err) {
        console.error('Failed to load settings rules:', err);
    }
}

function loadSdSettings() {
    const host = localStorage.getItem('vetinari_sd_host') || 'http://localhost:7860';
    const el = document.getElementById('sdHostInput');
    if (el) el.value = host;
}

// ──────────────────────────────────────────────────────────────────────────────
// Initialize on page load
// ──────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Detect hardware for intake form
    detectHardware();

    // Populate model selects in intake form
    setTimeout(async () => {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            const models = data.models || data;
            if (Array.isArray(models)) {
                const selects = [
                    document.getElementById('intakeModelSelect'),
                    document.getElementById('modelRulesModelSelect'),
                    document.getElementById('trainingModelSelect'),
                ];
                selects.forEach(sel => {
                    if (!sel) return;
                    models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m.id || m.name || m;
                        opt.textContent = m.name || m.id || m;
                        sel.appendChild(opt);
                    });
                });
            }
        } catch (e) { /* ignore */ }
    }, 1000);

    // Show intake form if no project selected on prompt view
    if (window.location.hash === '#prompt' || document.querySelector('.nav-item.active[data-view="prompt"]')) {
        const form = document.getElementById('projectIntakeForm');
        const chatInputArea = document.getElementById('chatInputArea');
        if (form && !window.currentProjectId) {
            form.style.display = 'flex';
            if (chatInputArea) chatInputArea.style.display = 'none';
        }
    }
});

// Show intake form by default when prompt view is first loaded
const promptViewEl = document.getElementById('promptView');
if (promptViewEl) {
    const observer = new MutationObserver(() => {
        if (promptViewEl.classList.contains('active') && !window.currentProjectId) {
            const form = document.getElementById('projectIntakeForm');
            const chatInputArea = document.getElementById('chatInputArea');
            if (form) form.style.display = 'flex';
            if (chatInputArea) chatInputArea.style.display = 'none';
            detectHardware();
        }
    });
    observer.observe(promptViewEl, { attributes: true, attributeFilter: ['class'] });
}

console.log('[Vetinari] Extensions loaded: intake form, markdown, task list, goal verification, rules, image gen');
