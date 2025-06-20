/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;
/* Find the L3 size by running `getconf -a | grep CACHE` */
const LLCSIZE = 16 * 1024 * 1024; // 16MB L3 cache from getconf
/* Collect traces for 10 seconds; you can vary this */
const TIME = 10000;
/* Collect traces every 10ms; you can vary this */
const P = 10; 

function sweep(P) {
    /*
     * Implement this function to run a sweep of the cache.
     * 1. Allocate a buffer of size LLCSIZE.
     * 2. Read each cache line (read the buffer in steps of LINESIZE).
     * 3. Count the number of times each cache line is read in a time period of P milliseconds.
     * 4. Store the count in an array of size K, where K = TIME / P.
     * 5. Return the array of counts.
     */
    
    // Allocate buffer of LLC size
    const buffer = new Uint8Array(LLCSIZE);
    
    // Initialize buffer to ensure it's allocated
    for (let i = 0; i < LLCSIZE; i += LINESIZE) {
        buffer[i] = 1;
    }
    
    const K = Math.floor(TIME / P);
    const counts = [];
    
    for (let k = 0; k < K; k++) {
        const startTime = performance.now();
        let sweepCount = 0;
        
        // Keep sweeping until P milliseconds have passed
        while ((performance.now() - startTime) < P) {
            // Read through entire buffer at cache line intervals
            let sum = 0;
            for (let i = 0; i < LLCSIZE; i += LINESIZE) {
                sum += buffer[i];
            }
            sweepCount++;
            
            // Prevent optimization
            if (sum < 0) console.log("impossible");
        }
        
        counts.push(sweepCount);
    }
    
    return counts;
}   

self.addEventListener('message', function(e) {
    /* Call the sweep function and return the result */
    if (e.data === 'start') {
        try {
            const traceData = sweep(P);
            self.postMessage({
                success: true,
                data: traceData,
                timestamp: Date.now()
            });
        } catch (error) {
            self.postMessage({
                success: false,
                error: error.message
            });
        }
    }
});