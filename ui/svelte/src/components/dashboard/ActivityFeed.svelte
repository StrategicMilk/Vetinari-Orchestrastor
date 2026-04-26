<script>
  /**
   * Recent activity feed for the dashboard.
   *
   * Shows a scrollable list of recent system events with timestamps,
   * icons, and severity-based styling.
   *
   * @prop {Array<{type: string, message: string, timestamp: string}>} events
   */
  import { relativeTime } from '$lib/utils/format.js';

  let { events = [] } = $props();

  const TYPE_ICONS = {
    task: 'fas fa-check-circle',
    error: 'fas fa-exclamation-circle',
    model: 'fas fa-microchip',
    training: 'fas fa-graduation-cap',
    system: 'fas fa-cog',
    project: 'fas fa-folder',
  };

  const TYPE_COLORS = {
    task: 'var(--success)',
    error: 'var(--danger)',
    model: 'var(--primary)',
    training: 'var(--secondary)',
    system: 'var(--text-muted)',
    project: 'var(--warning)',
  };

  function iconFor(type) {
    return TYPE_ICONS[type] ?? 'fas fa-info-circle';
  }

  function colorFor(type) {
    return TYPE_COLORS[type] ?? 'var(--text-muted)';
  }
</script>

<div class="activity-feed">
  <h3 class="feed-title">
    <i class="fas fa-stream"></i>
    Recent Activity
  </h3>

  {#if events.length === 0}
    <p class="feed-empty">No recent activity</p>
  {:else}
    <div class="feed-list">
      {#each events.slice(0, 20) as event}
        <div class="feed-item">
          <div class="feed-icon" style="color: {colorFor(event.type)}">
            <i class={iconFor(event.type)}></i>
          </div>
          <div class="feed-body">
            <span class="feed-message">{event.message}</span>
            <span class="feed-time">{relativeTime(event.timestamp)}</span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .activity-feed {
    padding: 20px;
    background: var(--surface-bg);
    border: 1px solid var(--border-default);
    border-radius: 12px;
  }

  .feed-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .feed-title i {
    color: var(--text-muted);
  }

  .feed-empty {
    color: var(--text-muted);
    font-size: 0.875rem;
    text-align: center;
    padding: 24px 0;
  }

  .feed-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 400px;
    overflow-y: auto;
  }

  .feed-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 4px;
    border-radius: 6px;
    transition: background 100ms;
  }

  .feed-item:hover {
    background: rgba(255, 255, 255, 0.03);
  }

  .feed-icon {
    width: 20px;
    text-align: center;
    font-size: 0.8125rem;
    padding-top: 2px;
    flex-shrink: 0;
  }

  .feed-body {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .feed-message {
    font-size: 0.8125rem;
    color: var(--text-primary);
    line-height: 1.4;
  }

  .feed-time {
    font-size: 0.6875rem;
    color: var(--text-muted);
  }
</style>
