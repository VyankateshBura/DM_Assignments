import React, { useState,useEffect } from 'react';
import { Paper, Grid, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import DisplayData from './DisplayData';

const CorrelationChart = ({ data }) => {
  const [xAxis, setXAxis] = useState('');
  const [yAxis, setYAxis] = useState('');
  const [covariance, setCovariance] = useState(0)
  const [status, setStatus] = useState("")
  const [PC, setPC] = useState(0)
  const handleXAxisChange = (event) => {
    setXAxis(event.target.value);
  };


  const attributeName = Object.keys(data[0]);
  // console.log(attributeName,xAxis,yAxis)
  const handleYAxisChange = (event) => {
    setYAxis(event.target.value);
  };
 
  useEffect(() => {   
    const fetchData = async () => {
      const requestData = {
        arrayData: data,
        attribute1: xAxis,
        attribute2: yAxis
      };

      try {
        const response = await axios.post(`http://127.0.0.1:8000/api/v1/correlation/`, requestData);
        // console.log("Response from backend ",response.data)
        if(response.data!=null){
          // console.log(response.data.data,response.data.data[3],response.data.data[4])
          setCovariance(response.data.data[3].data)
          setStatus(response.data.data[4].data)
          setPC(response.data.data[2].data)
        }
        
      } catch (error) {
        console.error('Error sending POST request:', error);
      }
    };
    if (xAxis && yAxis) {
        fetchData();
    }
  }, [xAxis,yAxis])
  
  
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
      {status!=""? (
      <DisplayData
        covariance={covariance}
        PC={PC}
        relationshipStatus={status}
      />
    ) : (
      <p>No contingency table data available.</p>
    )}
      {/* <LineChart
        width={800}
        height={400}
        data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="y" stroke="#8884d8" />
      </LineChart> */}
    </Paper>
  );
};

export default CorrelationChart;
