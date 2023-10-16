import React, { useState,useEffect } from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
// import ScatterPlotChart from '../../ScatterPlot/ScatterPlotChart';
import Dendogram from "./dendrogram.png"

import ReactPaginate from 'react-paginate';

const AGNES = ({ data ,url,title}) => {
  

 
  useEffect(() => {
    const fetchData = async () => {
      const requestData = {
        arrayData: data,
        var1: "Hi",
        var2: "Hello"
      };

      try {
        const response = await axios.post(`http://127.0.0.1:8000/api/v1/${url}/`, requestData);
        console.log("Response from backend ",response)

      } catch (error) {
        console.error('Error sending POST request:', error);
      }
    };

    if (data.length > 0) {
      fetchData();
    }
  }, [data]);


  

  return (
    <div>
        { Dendogram && <img src={Dendogram} alt="Loading please wait"/>}
    </div>
  );
};

export default AGNES;
