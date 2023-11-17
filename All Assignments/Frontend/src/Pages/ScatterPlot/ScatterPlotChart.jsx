import React, { useState } from 'react';
import { Paper, Grid, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import Papa from 'papaparse';

const ScatterPlotChart = ({ data }) => {
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
      <ScatterChart
        width={800}
        height={400}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid />
        <XAxis type="number" dataKey="x" name={xAxis} />
        <YAxis type="number" dataKey="y" name={yAxis} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
        <Legend />
        <Scatter name="Data" data={chartData} fill="#8884d8" />
      </ScatterChart>
    </Paper>
  );
};

export default ScatterPlotChart;
