import React, { useState,useEffect } from 'react';
import { Paper, Grid, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import ContingencyTable from './ContingencyTable';

const ChiSquareChart = ({ data }) => {
  const [xAxis, setXAxis] = useState('');
  const [yAxis, setYAxis] = useState('');
  const [ChiData,setChiData] = useState({});
  const [contingency_table, setContingency_table] = useState([])
  const [dof, setDof] = useState(0)
  const [p, setP] = useState(0)
  const [status, setStatus] = useState("")
  const [chi2, setChi2] = useState(0)
  const handleXAxisChange = (event) => {
    setXAxis(event.target.value);
  };


  const attributeName = Object.keys(data[0]);
  console.log(attributeName,xAxis,yAxis)
  const handleYAxisChange = (event) => {
    setYAxis(event.target.value);
  };
 
  useEffect(() => {   
    const fetchData = async () => {
      const requestData = {
        arrayData: data,
        col1: xAxis,
        col2: yAxis
      };

      try {
        const response = await axios.post(`http://127.0.0.1:8000/api/v1/chisquare/`, requestData);
        console.log("Response from backend ",response.data)
        if(response.data!=null){
          console.log(response.data)
          setContingency_table(response.data.contingency_table)
          setP(response.data.p)
          setDof(response.data.dof)
          setStatus(response.data.status)
          setChi2(response.data.chi2)
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
  // console.log(ChiData.contingency_table,ChiData.chi2,ChiData.p,ChiData.dof,ChiData.expected,ChiData.status)
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
      {contingency_table!=[]? (
      <ContingencyTable
        data={contingency_table}
        chi2={chi2}
        dof={dof}
        p={p}
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

export default ChiSquareChart;
