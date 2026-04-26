<script>
  /**
   * Model recommendations panel.
   *
   * Shows recommended models based on scoring and current task context.
   *
   * @prop {Array<object>} recommendations - Scored model recommendations.
   * @prop {(modelId: string) => void} [onselect] - Called when a model is selected.
   */
  let { recommendations = [], onselect } = $props();
</script>

{#if recommendations.length > 0}
  <div class="recommendations-panel">
    <h3 class="rec-title">
      <i class="fas fa-star"></i>
      Recommended Models
    </h3>

    <div class="rec-list">
      {#each recommendations.slice(0, 5) as rec}
        <button class="rec-item" onclick={() => onselect?.(rec.id ?? rec.name)}>
          <div class="rec-name">{rec.name ?? rec.id}</div>
          <div class="rec-meta">
            {#if rec.recommended_for?.length}
              <span class="rec-reason">{rec.recommended_for.join(', ')}</span>
            {:else if rec.reason}
              <span class="rec-reason">{rec.reason}</span>
            {/if}
            {#if rec.memory_gb}
              <span class="rec-size">{rec.memory_gb} GB</span>
            {/if}
          </div>
        </button>
      {/each}
    </div>
  </div>
{/if}

<style>
  .recommendations-panel {
    padding: 20px;
    background: var(--surface-bg);
    border: 1px solid var(--border-default);
    border-radius: 12px;
  }

  .rec-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .rec-title i { color: var(--warning); }

  .rec-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .rec-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    color: var(--text-primary);
    text-align: left;
    font-family: inherit;
    cursor: pointer;
    border-radius: 8px;
    transition: background 100ms;
    width: 100%;
  }

  .rec-item:hover {
    background: rgba(255, 255, 255, 0.04);
  }

  .rec-name {
    font-size: 0.875rem;
    font-weight: 500;
  }

  .rec-meta {
    display: flex;
    gap: 10px;
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .rec-size {
    color: var(--text-muted);
    font-weight: 400;
  }
</style>
