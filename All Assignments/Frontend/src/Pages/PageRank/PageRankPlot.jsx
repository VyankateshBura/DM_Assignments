import React, { useState, useEffect } from 'react';
import { Paper, TableContainer, TableHead, TableRow, TableCell, TableBody, Grid } from '@material-ui/core';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { Table } from 'antd';
import { DataGrid, GridToolbar } from "@mui/x-data-grid";
// import KMedoidsPlot from "./kmedoids_clusters.png"
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import ReactPaginate from 'react-paginate';

const PageRankPlot = ({ data, url, title }) => {

    const [fetchedData, setFetchedData] = useState(null)
    const [loading, setLoading] = useState(false);
   




    useEffect(() => {
        setLoading(true);
        const fetchData = async () => {
            const requestData = {
                arrayData: data, // Use the selected value from the dropdown
               
            };
            
            try {
                const response = await axios.post(`http://127.0.0.1:8000/api/v1/calculate_pagerank/`, requestData);
               
                console.log("Response from backend ", response)
                setFetchedData(response.data);
               
                
            } catch (error) {
                console.error('Error sending POST request:', error);
            }
        };
        fetchData();
        setLoading(false);
    }, [data,url]);

    // Function to extract row id
    console.log("Fetched data ",fetchedData)
    return (
        <>
        {/* {loading ? (
                <p>Loading...</p>
            ) : ( */}
                <div>
                    <h1>Top 10 Pages </h1>
                    <Grid container spacing={2}>
        {fetchedData &&
          fetchedData.rank_table.map((row, index) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={index}>
              <Paper>
                <p>{row.Page}</p>
                <p>{row.Rank}</p>
              </Paper>
            </Grid>
          ))}
      </Grid>
                </div>
            {/* )} */}
        </>
    );
};

export default PageRankPlot;
