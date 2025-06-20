function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // ML Model state
    modelStatus: {
      model_loaded: false,
      available_websites: [],
      model_type: null
    },
    latestPrediction: null,
    latestTrace: null,

    // Initialize by fetching model status
    async init() {
      await this.fetchModelStatus();
      await this.fetchResults();
    },

    // Fetch ML model status from backend
    async fetchModelStatus() {
      try {
        const response = await fetch('/api/model_status');
        if (response.ok) {
          this.modelStatus = await response.json();
        }
      } catch (error) {
        console.error("Error fetching model status:", error);
      }
    },

    // Collect trace data with real-time prediction
    async collectTraceWithPrediction() {
      this.isCollecting = true;
      this.status = "Collecting trace data with ML prediction... This will take about 10 seconds.";
      this.statusIsError = false;
      this.showingTraces = true;

      try {
        // Create worker
        let worker = new Worker("worker.js");

        // Start trace collection and wait for result
        const result = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        if (!result.success) {
          throw new Error(result.error);
        }

        // Store latest trace for potential re-prediction
        this.latestTrace = result.data;

        // Send trace data to backend with prediction
        const response = await fetch('/collect_trace_with_prediction', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            trace: result.data,
            timestamp: result.timestamp
          })
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const backendResult = await response.json();
        
        // Add heatmap to local collection
        this.heatmaps.push({
          path: backendResult.heatmap_path,
          timestamp: result.timestamp,
          dataPoints: result.data.length,
          prediction: backendResult.prediction
        });
        
        // Store trace data locally
        this.traceData.push({
          data: result.data,
          timestamp: result.timestamp,
          prediction: backendResult.prediction
        });

        // Update latest prediction
        if (backendResult.prediction) {
          this.latestPrediction = {
            ...backendResult.prediction,
            timestamp: result.timestamp
          };
          this.status = `Trace collected and analyzed! Predicted: ${backendResult.prediction.website} (${(backendResult.prediction.confidence * 100).toFixed(1)}% confidence)`;
        } else {
          this.status = `Trace collection complete! Collected ${result.data.length} data points. (No prediction available)`;
        }
        
        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting trace data with prediction:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Predict website from the latest collected trace
    async predictFromLastTrace() {
      if (!this.latestTrace) {
        this.status = "No trace data available for prediction";
        this.statusIsError = true;
        return;
      }

      this.status = "Making prediction from latest trace...";
      this.statusIsError = false;

      try {
        const response = await fetch('/api/predict_website', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            trace: this.latestTrace
          })
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.success) {
          this.latestPrediction = result.prediction;
          this.status = `Prediction: ${result.prediction.website} (${(result.prediction.confidence * 100).toFixed(1)}% confidence)`;
        } else {
          throw new Error(result.error);
        }
      } catch (error) {
        console.error("Error making prediction:", error);
        this.status = `Prediction error: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
       /* 
        * Implement this function to collect trace data.
        * 1. Create a worker to run the sweep function.
        * 2. Collect the trace data from the worker.
        * 3. Send the trace data to the backend for temporary storage and heatmap generation.
        * 4. Fetch the heatmap from the backend and add it to the local collection.
        * 5. Handle errors and update the status.
        */
        
        this.isCollecting = true;
        this.status = "Collecting trace data... This will take about 10 seconds.";
        this.statusIsError = false;
        this.showingTraces = true;

        try {
            // Create worker
            let worker = new Worker("worker.js");

            // Start trace collection and wait for result
            const result = await new Promise((resolve) => {
                worker.onmessage = (e) => resolve(e.data);
                worker.postMessage("start");
            });

            if (!result.success) {
                throw new Error(result.error);
            }

            // Send trace data to backend
            const response = await fetch('/collect_trace', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    trace: result.data,
                    timestamp: result.timestamp
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const backendResult = await response.json();
            
            // Add heatmap to local collection
            this.heatmaps.push({
                path: backendResult.heatmap_path,
                timestamp: result.timestamp,
                dataPoints: result.data.length
            });
            
            // Store trace data locally
            this.traceData.push({
                data: result.data,
                timestamp: result.timestamp
            });

            this.status = `Trace collection complete! Collected ${result.data.length} data points.`;
            
            // Terminate worker
            worker.terminate();
        } catch (error) {
            console.error("Error collecting trace data:", error);
            this.status = `Error: ${error.message}`;
            this.statusIsError = true;
        } finally {
            this.isCollecting = false;
        }
    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
       /* 
        * Implement this function to download the trace data.
        * 1. Fetch the latest data from the backend API.
        * 2. Create a download file with the trace data in JSON format.
        * 3. Handle errors and update the status.
        */
        
        try {
            const response = await fetch('/api/download_traces');
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Create download file
            const dataStr = JSON.stringify(data, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `traces_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            this.status = `Downloaded ${data.traces ? data.traces.length : 0} traces`;
        } catch (error) {
            console.error("Error downloading traces:", error);
            this.status = `Download error: ${error.message}`;
            this.statusIsError = true;
        }
    },

    // Clear all results from the server
    async clearResults() {
      /* 
       * Implement this function to clear all results from the server.
       * 1. Send a request to the backend API to clear all results.
       * 2. Clear local copies of trace data and heatmaps.
       * 3. Handle errors and update the status.
       */
      
      try {
          const response = await fetch('/api/clear_results', {
              method: 'POST'
          });
          
          if (!response.ok) {
              throw new Error(`Server error: ${response.status}`);
          }
          
          // Clear local data
          this.traceData = [];
          this.heatmaps = [];
          this.latencyResults = null;
          this.showingTraces = false;
          this.latestPrediction = null;
          this.latestTrace = null;
          
          this.status = "All results cleared successfully";
      } catch (error) {
          console.error("Error clearing results:", error);
          this.status = `Clear error: ${error.message}`;
          this.statusIsError = true;
      }
    },
    
    // Collect advanced trace data using multiple techniques
    async collectAdvancedTrace() {
        this.isCollecting = true;
        this.status = "Collecting advanced trace data... This will take about 10 seconds.";
        this.statusIsError = false;
        this.showingTraces = true;

        try {
            // Create advanced worker
            let worker = new Worker("advanced_worker.js");

            // Start advanced trace collection
            const result = await new Promise((resolve) => {
                worker.onmessage = (e) => resolve(e.data);
                worker.postMessage({
                    command: "advanced_sweep",
                    technique: "combined"  // Use combined advanced techniques
                });
            });

            if (!result.success) {
                throw new Error(result.error);
            }

            // Send advanced trace data to backend
            const response = await fetch('/collect_advanced_trace', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    trace: result.data,
                    technique: result.technique,
                    metadata: result.metadata,
                    timestamp: result.timestamp
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const backendResult = await response.json();
            
            // Add advanced heatmap to local collection
            this.heatmaps.push({
                path: backendResult.heatmap_path,
                timestamp: result.timestamp,
                dataPoints: Array.isArray(result.data) ? result.data.length : "Advanced",
                technique: result.technique
            });
            
            // Store advanced trace data locally
            this.traceData.push({
                data: result.data,
                timestamp: result.timestamp,
                technique: result.technique,
                metadata: result.metadata
            });

            this.status = `Advanced trace collection complete! Used technique: ${result.technique}`;
            
            // Terminate worker
            worker.terminate();
        } catch (error) {
            console.error("Error collecting advanced trace data:", error);
            this.status = `Error: ${error.message}`;
            this.statusIsError = true;
        } finally {
            this.isCollecting = false;
        }
    },
    
    // Fetch existing results when page loads
    async fetchResults() {
        try {
            const response = await fetch('/api/get_results');
            if (response.ok) {
                const data = await response.json();
                this.heatmaps = data.heatmaps || [];
                this.traceData = data.traces || [];
            }
        } catch (error) {
            console.log("No existing results to load");
        }
    }
  };
}
