import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Paper, Grid, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import Papa from 'papaparse';
import { quantile } from 'd3-array'; // Import the quantile function from d3-array
import { quantileSorted } from 'simple-statistics';

// Function to generate random data from a normal distribution
function generateRandomNormal(mean, stdDev, n) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    data.push(mean + stdDev * z0);
  }
  return data;
}

const QuantilePlotChart = ({ data }) => {
  const [xAxis, setXAxis] = useState('');
  const [yAxis, setYAxis] = useState('');
  const [qqData, setQqData] = useState([]);

  useEffect(() => {
    if (xAxis && yAxis && data) {
      const xColumn = data.map(row => parseFloat(row[xAxis]));
      const yColumn = data.map(row => parseFloat(row[yAxis]));

      const sortedX = xColumn.sort((a, b) => a - b);
      const sortedY = yColumn.sort((a, b) => a - b);

      const quantiles = Array.from({ length: sortedX.length }, (_, i) => i / (sortedX.length - 1));
      const quantileX = quantiles.map(q => quantileSorted(sortedX, q));
      const quantileY = quantiles.map(q => quantileSorted(sortedY, q));

      const qqData = quantiles.map((q, index) => ({
        x: quantileX[index],
        y: quantileY[index],
      }));

      setQqData(qqData);

     
    }
  }, [xAxis, yAxis, data]);

  const handleXAxisChange = (event) => {
    setXAxis(event.target.value);
  };

  const handleYAxisChange = (event) => {
    setYAxis(event.target.value);
  };


  return (
    <Paper style={{ padding: '20px' }}>
    <Grid container spacing={2} alignItems="center">
      <Grid item>
        <FormControl variant="outlined">
          <InputLabel htmlFor="x-axis-select">X-Axis</InputLabel>
          <Select
            value={xAxis}
            onChange={handleXAxisChange}
            label="X-Axis"
            inputProps={{
              name: 'x-axis',
              id: 'x-axis-select',
            }}
          >
            {data && Object.keys(data[0]).map((key) => (
              <MenuItem key={key} value={key}>
                {key}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      <Grid item>
        <FormControl variant="outlined">
          <InputLabel htmlFor="y-axis-select">Y-Axis</InputLabel>
          <Select
            value={yAxis}
            onChange={handleYAxisChange}
            label="Y-Axis"
            inputProps={{
              name: 'y-axis',
              id: 'y-axis-select',
            }}
          >
            {data && Object.keys(data[0]).map((key) => (
              <MenuItem key={key} value={key}>
                {key}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
    </Grid>
    <Plot
      data={[
        {
          type: 'scatter',
          x: qqData.map((point) => point.x),
          y: qqData.map((point) => point.y),
          mode: 'markers',
          marker: { color: 'blue' },
          name: 'QQ Plot',
        },
      ]}
      layout={{
        title: 'Quantile-Quantile (QQ) Plot',
        xaxis: { title: xAxis },
        yaxis: { title: yAxis },
      }}
    />
  </Paper>
  );
};

export default QuantilePlotChart;
