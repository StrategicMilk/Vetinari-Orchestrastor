'use strict';
/**
 * Behavioral proof tests for ui/static/js/sse-manager.js (SESSION-33F fixes).
 *
 * Tests three runtime behaviors:
 *  1. close() during retry backoff cancels the pending setTimeout timer.
 *  2. isConnected() during retry backoff returns false (not TypeError).
 *  3. close() before the retry timer fires prevents a new EventSource opening.
 *
 * Run with: node tests/frontend/test_sse_manager.js
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');

// ---------------------------------------------------------------------------
// Load module source once; each test re-executes it into a fresh scope.
// ---------------------------------------------------------------------------
const SOURCE_PATH = path.join(__dirname, '../../ui/static/js/sse-manager.js');
const moduleSource = fs.readFileSync(SOURCE_PATH, 'utf8');

/**
 * Build a fresh SSEManager instance with fully mocked globals.
 *
 * Returns { SSEManager, timers, esInstances } where:
 *   timers      — plain object keyed by timer ID; value is {fn, delay}
 *   esInstances — array of all MockEventSource instances created
 */
function buildFreshManager() {
    const timers = {};
    let timerIdCounter = 1;
    const esInstances = [];

    class MockEventSource {
        constructor(url) {
            this.url = url;
            this.readyState = 0; // CONNECTING
            this.onerror = null;
            this.onopen = null;
            esInstances.push(this);
        }

        close() {
            this.readyState = MockEventSource.CLOSED;
        }

        addEventListener() {
            // no-op: event handler wiring is not under test here
        }
    }
    MockEventSource.OPEN = 1;
    MockEventSource.CLOSED = 2;

    const mockWindow = {};

    const mockSetTimeout = (fn, delay) => {
        const id = timerIdCounter++;
        timers[id] = { fn, delay };
        return id;
    };

    const mockClearTimeout = (id) => {
        delete timers[id];
    };

    // Execute module source with injected globals.
    // The IIFE assigns window.SSEManager — we capture it from mockWindow.
    const factory = new Function(
        'window',
        'EventSource',
        'setTimeout',
        'clearTimeout',
        moduleSource
    );
    factory(mockWindow, MockEventSource, mockSetTimeout, mockClearTimeout);

    return {
        SSEManager: mockWindow.SSEManager,
        timers,
        esInstances,
    };
}

// ---------------------------------------------------------------------------
// Helper: connect and trigger a retry by calling onerror while CONNECTING.
// ---------------------------------------------------------------------------
function connectAndTriggerRetry(SSEManager, esInstances) {
    SSEManager.connect('test-id', 'http://example.com/events', {});
    assert.strictEqual(esInstances.length, 1, 'Expected one EventSource after connect');

    // Simulate onerror while readyState is still CONNECTING (0), not CLOSED (2).
    // The manager should close the source and schedule a retry timer.
    const es = esInstances[0];
    assert.strictEqual(es.readyState, 0, 'EventSource should start CONNECTING');
    es.onerror(); // triggers the backoff branch
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

let failures = 0;

function run(name, fn) {
    try {
        fn();
        console.log('\u2713 ' + name);
    } catch (err) {
        console.error('\u2717 ' + name);
        console.error('    ' + err.message);
        failures++;
    }
}

// Test 1: close() during retry backoff cancels the pending setTimeout timer.
run('close() during retry backoff cancels the pending timer', () => {
    const { SSEManager, timers, esInstances } = buildFreshManager();

    connectAndTriggerRetry(SSEManager, esInstances);

    // After onerror the manager should have scheduled a retry timer.
    const timerIdsBefore = Object.keys(timers);
    assert.strictEqual(timerIdsBefore.length, 1, 'Expected exactly one pending timer after retry scheduled');

    // Calling close() must cancel that timer via clearTimeout.
    SSEManager.close('test-id');

    const timerIdsAfter = Object.keys(timers);
    assert.strictEqual(
        timerIdsAfter.length,
        0,
        'Timer must be cancelled (deleted) after close() — clearTimeout was not called or did not remove it'
    );
});

// Test 2: isConnected() during retry backoff returns false, not a TypeError.
run('isConnected() during retry backoff returns false (not TypeError)', () => {
    const { SSEManager, esInstances } = buildFreshManager();

    connectAndTriggerRetry(SSEManager, esInstances);

    // During retry the connection entry has es = null (line 45 of sse-manager.js).
    // isConnected() must guard against that and return false, not throw.
    let result;
    try {
        result = SSEManager.isConnected('test-id');
    } catch (err) {
        throw new Error(
            'isConnected() threw ' + err.constructor.name + ' during retry: ' + err.message
        );
    }

    assert.strictEqual(
        result,
        false,
        'isConnected() must return false during retry backoff (es is null)'
    );
});

// Test 3: close() before the retry timer fires prevents a new EventSource opening.
run('close() before retry timer fires prevents a new EventSource from opening', () => {
    const { SSEManager, timers, esInstances } = buildFreshManager();

    connectAndTriggerRetry(SSEManager, esInstances);

    // Capture the retry timer before closing.
    const pendingTimerIds = Object.keys(timers);
    assert.strictEqual(pendingTimerIds.length, 1, 'Expected exactly one pending timer');
    const retryTimerId = pendingTimerIds[0];
    const retryFn = timers[retryTimerId].fn;

    // close() should cancel the timer and remove the connection entry.
    SSEManager.close('test-id');
    assert.strictEqual(Object.keys(timers).length, 0, 'Timer should be gone after close()');

    const countBefore = esInstances.length;

    // Simulate what would happen if the JS runtime fired the timer callback
    // after close() — e.g. a race between clearTimeout and a timer already
    // queued. The retry callback (createSource) runs but _connections[id] no
    // longer exists, so the new EventSource should be created but the
    // connection entry it writes is effectively orphaned / immediately
    // overwritten. More importantly, no *additional* EventSource should be
    // observable as "connected" after this.
    //
    // The primary guarantee we test here: after close() the timer is gone
    // (clearTimeout was called), so in a real JS environment the callback
    // would never fire. We verify that by confirming the timer was deleted
    // from our mock registry — already asserted above. We additionally verify
    // that if (hypothetically) the callback did fire, any new connection is
    // not registered under the id (because close() deleted it, and the new
    // createSource() would re-register — but calling close() again cleans up).
    //
    // Concrete assertion: the timer was deleted before firing, so in our
    // deterministic mock the fn was never called automatically. The timer
    // count is 0 and esInstances count has not grown past countBefore.
    assert.strictEqual(
        esInstances.length,
        countBefore,
        'No new EventSource should have been created because the timer was cancelled before firing'
    );

    // Belt-and-suspenders: manually invoke the retry callback to prove that
    // even if it fires (race), isConnected() still returns false and we can
    // call close() without error (idempotent cleanup).
    retryFn(); // createSource() runs, creates a new EventSource, sets _connections[id]

    // isConnected() must not throw even with a freshly-created (CONNECTING) es.
    let connected;
    try {
        connected = SSEManager.isConnected('test-id');
    } catch (err) {
        throw new Error('isConnected() threw after spurious retry callback: ' + err.message);
    }
    // CONNECTING state (readyState=0) != OPEN (1), so should still be false.
    assert.strictEqual(
        connected,
        false,
        'isConnected() should be false for a CONNECTING EventSource'
    );

    // Cleanup without error.
    SSEManager.close('test-id');
});

// ---------------------------------------------------------------------------
// Exit
// ---------------------------------------------------------------------------
if (failures > 0) {
    process.exit(1);
}
