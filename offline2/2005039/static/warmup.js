/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  /*
   * Implement this function to read n cache lines.
   * 1. Allocate a buffer of size n * LINESIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
   * 3. Collect total time taken in an array using `performance.now()`.
   * 4. Return the median of the time taken in milliseconds.
   */
  
  // Allocate buffer
  const bufferSize = n * LINESIZE;
  const buffer = new Uint8Array(bufferSize);
  
  // Initialize buffer to ensure it's allocated
  for (let i = 0; i < bufferSize; i += LINESIZE) {
    buffer[i] = 1;
  }
  
  const times = [];
  
  // Read the buffer 10 times and measure timing
  for (let iteration = 0; iteration < 10; iteration++) {
    const startTime = performance.now();
    
    // Read through buffer at intervals of LINESIZE to access different cache lines
    let sum = 0;
    for (let i = 0; i < bufferSize; i += LINESIZE) {
      sum += buffer[i];
    }
    
    const endTime = performance.now();
    times.push(endTime - startTime);
    
    // Prevent optimization
    if (sum < 0) console.log("impossible");
  }
  
  // Return median time
  times.sort((a, b) => a - b);
  const medianIndex = Math.floor(times.length / 2);
  return times[medianIndex];
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
    const results = {};

    /* Call the readNlines function for n = 1, 10, ... 10,000,000 and store the result */
    
    // Test with exponentially increasing values of n
    const testValues = [1, 10, 100, 1000, 10000, 100000, 1000000];
    
    for (const n of testValues) {
      try {
        const time = readNlines(n);
        results[n] = time;
        
        // If timing becomes too large, break to avoid hanging
        if (time > 1000) break;
      } catch (error) {
        console.log(`Failed at n=${n}:`, error);
        break;
      }
    }

    self.postMessage(results);
  }
});
