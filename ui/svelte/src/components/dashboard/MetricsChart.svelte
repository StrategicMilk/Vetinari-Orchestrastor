<script>
  /**
   * Chart.js wrapper component for dashboard timeseries charts.
   *
   * Creates and manages a Chart.js instance using design tokens for styling.
   * Automatically destroys the chart on component teardown and updates
   * reactively when data changes.
   *
   * @prop {string} title - Chart heading text.
   * @prop {'line'|'bar'|'doughnut'} [type='line'] - Chart.js chart type.
   * @prop {object} data - Chart.js data configuration.
   * @prop {object} [options={}] - Chart.js options override.
   */
  import { Chart, registerables } from 'chart.js';
  import { getChartDefaults } from '$lib/tokens.js';

  // Register all Chart.js components once
  Chart.register(...registerables);

  let { title, type = 'line', data, options = {} } = $props();

  let canvasEl;
  let chartInstance = null;

  /**
   * Build merged Chart.js options from token defaults and user overrides.
   *
   * @param {string} chartType - The chart type ('line', 'bar', 'doughnut').
   * @returns {object} Merged options object ready for Chart.js.
   */
  function buildOptions(chartType) {
    const defaults = getChartDefaults(chartType);
    return {
      ...defaults.options,
      ...options,
      plugins: { ...defaults.options.plugins, ...options.plugins },
      scales: chartType === 'doughnut'
        ? {}
        : { ...defaults.options.scales, ...options.scales },
    };
  }

  /**
   * Create or update the Chart.js instance.
   *
   * When the `type` prop changes (e.g. 'line' → 'bar') the existing chart
   * instance is destroyed before creating a new one. Chart.js 4.x does allow
   * mutating config.type, but it requires separate scale registration and
   * re-initialisation that is error-prone. Destroy+recreate is the documented
   * safe path for type transitions.
   *
   * When only data or options change, the instance is updated in-place to
   * avoid the canvas flicker a full recreate would cause.
   */
  function renderChart() {
    if (!canvasEl || !data) return;

    const mergedOptions = buildOptions(type);

    // Destroy the existing instance when the chart type has changed so the
    // canvas is re-initialised with correct scale and element registrations.
    if (chartInstance && chartInstance.config.type !== type) {
      chartInstance.destroy();
      chartInstance = null;
    }

    if (chartInstance) {
      chartInstance.data = data;
      chartInstance.options = mergedOptions;
      chartInstance.update('none');
    } else {
      chartInstance = new Chart(canvasEl, {
        type,
        data,
        options: mergedOptions,
      });
    }
  }

  // Reactively re-render when data, options, or type changes.
  // All three are referenced explicitly so Svelte tracks each as a dependency.
  $effect(() => {
    void data;
    void options;
    void type;
    renderChart();
  });

  // Clean up on destroy
  $effect(() => {
    return () => {
      if (chartInstance) {
        chartInstance.destroy();
        chartInstance = null;
      }
    };
  });
</script>

<div class="metrics-chart">
  {#if title}
    <h3 class="chart-title">{title}</h3>
  {/if}
  <div class="chart-container">
    <canvas bind:this={canvasEl}></canvas>
  </div>
</div>

<style>
  .metrics-chart {
    padding: 20px;
    background: var(--surface-bg);
    border: 1px solid var(--border-default);
    border-radius: 12px;
  }

  .chart-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 12px 0;
  }

  .chart-container {
    position: relative;
    height: 200px;
  }
</style>
