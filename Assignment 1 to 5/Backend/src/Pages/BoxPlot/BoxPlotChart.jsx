import React, { useState } from 'react';
import { Paper, Grid, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import Plot from 'react-plotly.js';
import Papa from 'papaparse';

const BoxPlotChart = ({ data }) => {
  const [xAxis, setXAxis] = useState('');
  const [yAxis, setYAxis] = useState('');

  const handleXAxisChange = (event) => {
    setXAxis(event.target.value);
  };

  const handleYAxisChange = (event) => {
    setYAxis(event.target.value);
  };

  const chartData = data.map((row) => ({
    x: row[xAxis],
    y: row[yAxis],
  }));

  const plotData = [
    {
      type: 'box',
      x: chartData.map((item) => item.x),
      y: chartData.map((item) => item.y),
    },
  ];

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
              {Object.keys(data[0]).map((key) => (
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
              {Object.keys(data[0]).map((key) => (
                <MenuItem key={key} value={key}>
                  {key}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
      </Grid>
      <Plot
        data={plotData}
        layout={{
          width: 800,
          height: 400,
          title: 'Box Plot Chart',
          xaxis: { title: xAxis },
          yaxis: { title: yAxis },
        }}
      />
    </Paper>
  );
};

export default BoxPlotChart;
