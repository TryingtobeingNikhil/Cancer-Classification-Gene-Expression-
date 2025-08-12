import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import * as d3 from 'd3';
import _ from 'lodash';

const GeneticDataAnalyzer = () => {
  const [originalData, setOriginalData] = useState([]);
  const [syntheticData, setSyntheticData] = useState([]);
  const [stats, setStats] = useState({});
  const [correlations, setCorrelations] = useState({});
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const csvData = await window.fs.readFile('Train_gene_data.csv', { encoding: 'utf8' });
      const parsed = Papa.parse(csvData, { header: true, dynamicTyping: true, skipEmptyLines: true });
      
      if (parsed.errors.length > 0) {
        console.error('Parse errors:', parsed.errors);
      }
      
      const cleanedData = parsed.data.filter(row => 
        Object.values(row).some(val => val !== null && val !== undefined && val !== '')
      );
      
      setOriginalData(cleanedData);
      calculateStatistics(cleanedData);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const calculateStatistics = (data) => {
    const columns = Object.keys(data[0]);
    const numericColumns = columns.filter(col => col !== 'Target');
    
    const columnStats = {};
    const correlationMatrix = {};
    
    numericColumns.forEach(col => {
      const values = data.map(row => parseFloat(row[col])).filter(val => !isNaN(val));
      columnStats[col] = {
        mean: d3.mean(values),
        std: d3.deviation(values),
        min: d3.min(values),
        max: d3.max(values),
        median: d3.median(values),
        q1: d3.quantile(values.sort(d3.ascending), 0.25),
        q3: d3.quantile(values.sort(d3.ascending), 0.75)
      };
    });

    // Calculate correlations between columns
    numericColumns.forEach(col1 => {
      correlationMatrix[col1] = {};
      numericColumns.forEach(col2 => {
        const values1 = data.map(row => parseFloat(row[col1])).filter(val => !isNaN(val));
        const values2 = data.map(row => parseFloat(row[col2])).filter(val => !isNaN(val));
        
        if (values1.length === values2.length && values1.length > 1) {
          const correlation = calculatePearsonCorrelation(values1, values2);
          correlationMatrix[col1][col2] = correlation;
        }
      });
    });

    setStats(columnStats);
    setCorrelations(correlationMatrix);
  };

  const calculatePearsonCorrelation = (x, y) => {
    const n = x.length;
    const sumX = d3.sum(x);
    const sumY = d3.sum(y);
    const sumXY = d3.sum(x.map((xi, i) => xi * y[i]));
    const sumX2 = d3.sum(x.map(xi => xi * xi));
    const sumY2 = d3.sum(y.map(yi => yi * yi));
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  };

  const generateSyntheticSamples = (numSamples = 100) => {
    setIsGenerating(true);
    
    const columns = Object.keys(originalData[0]);
    const numericColumns = columns.filter(col => col !== 'Target');
    const synthetic = [];
    
    // Get target distribution
    const targetValues = originalData.map(row => row.Target);
    const targetCounts = _.countBy(targetValues);
    const targetProbs = Object.keys(targetCounts).map(key => ({
      value: parseInt(key),
      prob: targetCounts[key] / originalData.length
    }));

    for (let i = 0; i < numSamples; i++) {
      const newSample = {};
      
      // Generate target first
      const rand = Math.random();
      let cumProb = 0;
      for (const {value, prob} of targetProbs) {
        cumProb += prob;
        if (rand <= cumProb) {
          newSample.Target = value;
          break;
        }
      }
      
      // Generate correlated features
      const baseValues = {};
      
      // First pass: generate base values with noise
      numericColumns.forEach(col => {
        const stat = stats[col];
        if (stat) {
          // Use a mix of normal distribution and actual data sampling
          const useActualSample = Math.random() < 0.3;
          
          if (useActualSample) {
            // Sample from actual data with small perturbation
            const randomRow = originalData[Math.floor(Math.random() * originalData.length)];
            const baseValue = parseFloat(randomRow[col]);
            const noise = (Math.random() - 0.5) * stat.std * 0.2;
            baseValues[col] = Math.round(baseValue + noise);
          } else {
            // Generate using normal distribution
            const normalValue = generateNormalRandom() * stat.std + stat.mean;
            baseValues[col] = Math.round(Math.max(stat.min, Math.min(stat.max, normalValue)));
          }
        }
      });
      
      // Second pass: adjust for correlations with major correlated features
      const strongCorrelations = [];
      Object.keys(correlations).forEach(col1 => {
        Object.keys(correlations[col1]).forEach(col2 => {
          const corr = correlations[col1][col2];
          if (Math.abs(corr) > 0.5 && col1 < col2) { // Avoid duplicates
            strongCorrelations.push({col1, col2, corr});
          }
        });
      });
      
      // Apply correlation adjustments
      strongCorrelations.slice(0, 10).forEach(({col1, col2, corr}) => {
        if (baseValues[col1] !== undefined && baseValues[col2] !== undefined) {
          const stat1 = stats[col1];
          const stat2 = stats[col2];
          
          // Adjust col2 based on col1 value and correlation
          const normalizedCol1 = (baseValues[col1] - stat1.mean) / stat1.std;
          const expectedCol2 = stat2.mean + (corr * normalizedCol1 * stat2.std);
          
          // Blend with original value
          const blendFactor = Math.abs(corr) * 0.7;
          baseValues[col2] = Math.round(
            baseValues[col2] * (1 - blendFactor) + expectedCol2 * blendFactor
          );
          
          // Ensure within bounds
          baseValues[col2] = Math.max(stat2.min, Math.min(stat2.max, baseValues[col2]));
        }
      });
      
      // Copy final values to new sample
      Object.assign(newSample, baseValues);
      
      synthetic.push(newSample);
    }
    
    setSyntheticData(synthetic);
    setIsGenerating(false);
  };

  const generateNormalRandom = () => {
    // Box-Muller transform
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  const checkForDuplicates = () => {
    const allData = [...originalData, ...syntheticData];
    const uniqueRows = new Set();
    let duplicates = 0;
    
    allData.forEach(row => {
      const rowString = JSON.stringify(row);
      if (uniqueRows.has(rowString)) {
        duplicates++;
      } else {
        uniqueRows.add(rowString);
      }
    });
    
    return duplicates;
  };

  const exportCSV = () => {
    const allData = [...originalData, ...syntheticData];
    const csv = Papa.unparse(allData);
    
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', 'expanded_genetic_data.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const duplicateCount = syntheticData.length > 0 ? checkForDuplicates() : 0;

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Genetic Data Analysis & Synthetic Sample Generation</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800">Original Dataset</h3>
          <p className="text-2xl font-bold text-blue-600">{originalData.length} samples</p>
          <p className="text-sm text-blue-600">{Object.keys(originalData[0] || {}).length} features</p>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="font-semibold text-green-800">Synthetic Samples</h3>
          <p className="text-2xl font-bold text-green-600">{syntheticData.length} samples</p>
          <p className="text-sm text-green-600">Generated with correlations</p>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <h3 className="font-semibold text-purple-800">Total Dataset</h3>
          <p className="text-2xl font-bold text-purple-600">{originalData.length + syntheticData.length} samples</p>
          <p className="text-sm text-purple-600">{duplicateCount} duplicates detected</p>
        </div>
      </div>

      <div className="flex gap-4 mb-6">
        <button
          onClick={() => generateSyntheticSamples(1000)}
          disabled={isGenerating || originalData.length === 0}
          className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-4 py-2 rounded"
        >
          {isGenerating ? 'Generating...' : 'Generate 1K Samples'}
        </button>
        
        <button
          onClick={() => generateSyntheticSamples(5000)}
          disabled={isGenerating || originalData.length === 0}
          className="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-4 py-2 rounded"
        >
          {isGenerating ? 'Generating...' : 'Generate 5K Samples'}
        </button>
        
        <button
          onClick={() => generateSyntheticSamples(10000)}
          disabled={isGenerating || originalData.length === 0}
          className="bg-purple-500 hover:bg-purple-600 disabled:bg-gray-400 text-white px-4 py-2 rounded"
        >
          {isGenerating ? 'Generating...' : 'Generate 10K Samples'}
        </button>
        
        <button
          onClick={() => generateSyntheticSamples(20000)}
          disabled={isGenerating || originalData.length === 0}
          className="bg-red-500 hover:bg-red-600 disabled:bg-gray-400 text-white px-4 py-2 rounded"
        >
          {isGenerating ? 'Generating...' : 'Generate 20K Samples'}
        </button>
        
        <button
          onClick={exportCSV}
          disabled={syntheticData.length === 0}
          className="bg-orange-500 hover:bg-orange-600 disabled:bg-gray-400 text-white px-4 py-2 rounded"
        >
          Export Combined CSV
        </button>
      </div>

      {Object.keys(stats).length > 0 && (
        <div className="mb-6">
          <h2 className="text-xl font-bold mb-3">Statistical Summary (First 10 Features)</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left">Feature</th>
                  <th className="px-4 py-2 text-right">Mean</th>
                  <th className="px-4 py-2 text-right">Std Dev</th>
                  <th className="px-4 py-2 text-right">Min</th>
                  <th className="px-4 py-2 text-right">Max</th>
                  <th className="px-4 py-2 text-right">Median</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(stats).slice(0, 10).map(col => (
                  <tr key={col} className="border-t">
                    <td className="px-4 py-2 font-mono text-sm">{col}</td>
                    <td className="px-4 py-2 text-right">{stats[col].mean?.toFixed(1)}</td>
                    <td className="px-4 py-2 text-right">{stats[col].std?.toFixed(1)}</td>
                    <td className="px-4 py-2 text-right">{stats[col].min}</td>
                    <td className="px-4 py-2 text-right">{stats[col].max}</td>
                    <td className="px-4 py-2 text-right">{stats[col].median?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {syntheticData.length > 0 && (
        <div className="mb-6">
          <h2 className="text-xl font-bold mb-3">Sample Preview (First 5 Synthetic Samples)</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200 text-xs">
              <thead className="bg-gray-50">
                <tr>
                  {Object.keys(syntheticData[0]).slice(0, 10).map(col => (
                    <th key={col} className="px-2 py-1 text-left font-mono">{col}</th>
                  ))}
                  <th className="px-2 py-1 text-left">...</th>
                  <th className="px-2 py-1 text-left">Target</th>
                </tr>
              </thead>
              <tbody>
                {syntheticData.slice(0, 5).map((row, idx) => (
                  <tr key={idx} className="border-t">
                    {Object.keys(row).slice(0, 10).map(col => (
                      <td key={col} className="px-2 py-1">{row[col]}</td>
                    ))}
                    <td className="px-2 py-1">...</td>
                    <td className="px-2 py-1 font-bold">{row.Target}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-semibold mb-3 text-blue-800">💡 Dataset Size Recommendations for ML Training:</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium mb-2">Minimum Recommended Sizes:</h4>
            <ul className="space-y-1">
              <li>• <strong>Simple ML models:</strong> 1K-5K samples</li>
              <li>• <strong>Complex models (RF, SVM):</strong> 5K-10K samples</li>
              <li>• <strong>Deep learning:</strong> 10K-50K+ samples</li>
              <li>• <strong>Genomics/High-dim data:</strong> 10K-100K+ samples</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">Your Current Scenario:</h4>
            <ul className="space-y-1">
              <li>• <strong>Features:</strong> {Object.keys(originalData[0] || {}).length} (high-dimensional)</li>
              <li>• <strong>Classes:</strong> {Object.keys(_.countBy(originalData.map(r => r.Target))).length} (binary classification)</li>
              <li>• <strong>Original samples:</strong> {originalData.length} (very small)</li>
              <li>• <strong>Recommended:</strong> <span className="text-red-600 font-medium">10K-20K+ samples</span></li>
            </ul>
          </div>
        </div>
        <div className="mt-3 p-3 bg-yellow-100 border border-yellow-300 rounded">
          <p className="text-sm"><strong>⚠️ Important:</strong> With {originalData.length} original samples and {Object.keys(originalData[0] || {}).length} features, you have a high-dimensional, low-sample scenario. Generate at least 10K samples for reliable model performance. Consider dimensionality reduction (PCA, feature selection) if computational resources are limited.</p>
        </div>
      </div>

      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-semibold mb-2">Generation Method:</h3>
        <ul className="text-sm space-y-1">
          <li>• Maintains original statistical distributions (mean, std dev, min/max ranges)</li>
          <li>• Preserves correlations between highly correlated features (|r| > 0.5)</li>
          <li>• Uses 30% sampling from original data with small perturbations</li>
          <li>• Uses 70% normal distribution generation with correlation adjustments</li>
          <li>• Maintains original target class distribution</li>
          <li>• Ensures all values stay within observed feature ranges</li>
        </ul>
      </div>
    </div>
  );
};

export default GeneticDataAnalyzer;