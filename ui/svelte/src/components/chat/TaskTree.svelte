<script>
  /**
   * Nested task hierarchy tree with expand/collapse.
   *
   * Renders tasks recursively with status badges, indentation, and
   * toggleable subtasks. Limits initial depth to avoid deep nesting.
   *
   * @prop {Array<object>} tasks - Flat or nested task list.
   * @prop {number} [depth=0] - Current nesting depth (internal).
   * @prop {number} [maxDepth=3] - Maximum auto-expanded depth.
   */

  import TaskTree from './TaskTree.svelte';

  let { tasks = [], depth = 0, maxDepth = 3 } = $props();

  /**
   * Tracks nodes the user explicitly opened (added on expand click).
   * Nodes in this set are open regardless of depth.
   */
  let expandedIds = $state(new Set());

  /**
   * Tracks nodes the user explicitly collapsed (added on collapse click).
   * Nodes in this set stay closed even when depth < maxDepth would auto-expand.
   */
  let collapsedIds = $state(new Set());

  function toggle(taskId) {
    if (expandedIds.has(taskId)) {
      // User is collapsing — remove from expanded, add to explicitly collapsed.
      const nextExp = new Set(expandedIds);
      nextExp.delete(taskId);
      expandedIds = nextExp;

      const nextCol = new Set(collapsedIds);
      nextCol.add(taskId);
      collapsedIds = nextCol;
    } else {
      // User is expanding — add to expanded, remove from explicitly collapsed.
      const nextExp = new Set(expandedIds);
      nextExp.add(taskId);
      expandedIds = nextExp;

      const nextCol = new Set(collapsedIds);
      nextCol.delete(taskId);
      collapsedIds = nextCol;
    }
  }

  function isExpanded(taskId) {
    // Once the user has explicitly toggled a node (present in expandedIds or
    // removed from it via collapse), honour that choice at every depth.
    // For nodes the user has never touched, auto-expand up to maxDepth.
    if (expandedIds.has(taskId)) {
      // User explicitly opened it (or it was opened by default and not yet closed).
      return true;
    }
    // If the user has never interacted with this node (not in expandedIds at
    // all), fall back to the depth-based default.  But once toggle() removes
    // it, the node is absent from expandedIds AND the depth guard below must
    // not re-open it — so we track explicit collapses separately.
    return depth < maxDepth && !collapsedIds.has(taskId);
  }

  const STATUS_CLASSES = {
    completed: 'status-completed',
    in_progress: 'status-in-progress',
    pending: 'status-pending',
    failed: 'status-failed',
    skipped: 'status-skipped',
  };

  const STATUS_ICONS = {
    completed: 'fas fa-check-circle',
    in_progress: 'fas fa-spinner fa-spin',
    pending: 'fas fa-circle',
    failed: 'fas fa-times-circle',
    skipped: 'fas fa-forward',
  };
</script>

{#if tasks.length > 0}
  <div class="task-tree" style="--depth: {depth}">
    {#each tasks as task}
      {@const hasChildren = task.subtasks?.length > 0 || task.children?.length > 0}
      {@const children = task.subtasks ?? task.children ?? []}
      {@const expanded = isExpanded(task.id)}

      <div class="task-node">
        <div class="task-row">
          {#if hasChildren}
            <button
              class="task-toggle"
              onclick={() => toggle(task.id)}
              aria-expanded={expanded}
              aria-label={expanded ? 'Collapse' : 'Expand'}
            >
              <i class="fas" class:fa-chevron-right={!expanded} class:fa-chevron-down={expanded}></i>
            </button>
          {:else}
            <span class="task-toggle-spacer"></span>
          {/if}

          <span class="task-status {STATUS_CLASSES[task.status] ?? ''}">
            <i class={STATUS_ICONS[task.status] ?? 'fas fa-circle'}></i>
          </span>

          <span class="task-description">{task.description ?? task.name ?? task.id}</span>

          {#if task.agent_type}
            <span class="task-agent">{task.agent_type}</span>
          {/if}
        </div>

        {#if hasChildren && expanded}
          <TaskTree tasks={children} depth={depth + 1} {maxDepth} />
        {/if}
      </div>
    {/each}
  </div>
{/if}

<style>
  .task-tree {
    padding-left: calc(var(--depth, 0) * 20px);
  }

  .task-node {
    margin-bottom: 2px;
  }

  .task-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 8px;
    border-radius: 6px;
    transition: background 100ms;
  }

  .task-row:hover {
    background: rgba(255, 255, 255, 0.03);
  }

  .task-toggle {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 0.6875rem;
    flex-shrink: 0;
  }

  .task-toggle-spacer {
    width: 20px;
    flex-shrink: 0;
  }

  .task-status {
    font-size: 0.75rem;
    flex-shrink: 0;
    width: 16px;
    text-align: center;
  }

  .status-completed { color: var(--success); }
  .status-in-progress { color: var(--secondary); }
  .status-pending { color: var(--text-muted); }
  .status-failed { color: var(--danger); }
  .status-skipped { color: var(--text-muted); opacity: 0.5; }

  .task-description {
    font-size: 0.8125rem;
    color: var(--text-primary);
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .task-agent {
    font-size: 0.6875rem;
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    color: var(--text-muted);
    flex-shrink: 0;
  }
</style>
