import React, { useState, useEffect } from 'react';
import { Paper, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { Table } from 'antd';
import { DataGrid, GridToolbar } from "@mui/x-data-grid";
// import KMedoidsPlot from "./kmedoids_clusters.png"
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import ReactPaginate from 'react-paginate';

const HITSPlot = ({ data, url, title }) => {

    const [fetchedData, setFetchedData] = useState(null)
    const [loading, setLoading] = useState(false);
   




    useEffect(() => {
        setLoading(true);
        const fetchData = async () => {
            const requestData = {
                arrayData: data, // Use the selected value from the dropdown
               
            };
            
            try {
                const response = await axios.post(`http://127.0.0.1:8000/api/v1/calculate_hits/`, requestData);
               
                console.log("Response from backend ", response)
                
                setFetchedData(response.data)
                
            } catch (error) {
                console.error('Error sending POST request:', error);
            }
        };
        fetchData();
        setLoading(false);
    }, [data,url]);

    // Function to extract row id
    console.log(fetchedData)
    return (
        <>
        {loading ? (
                <p>Loading...</p>
            ) : (
                <div>
                    <h1>HITS </h1>
                    <h2>Adjacency Matrix:</h2>
          {/* <table border="1">
           
            {fetchedData.data.adjacency_matrix.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((value, colIndex) => (
                  <td key={colIndex}>{value}</td>
                ))}
              </tr>
            ))}
          </table> */}

          <h2>Top 10 Authorities:</h2>
          <ol>
            { fetchedData && fetchedData.data.authority_rank.map(([node, rank]) => (
              <li key={node}>Node {node} - Rank: {rank}</li>
            ))}
          </ol>

          <h2>Top 10 Hubs:</h2>
          <ol>
            { fetchedData && fetchedData.data.hub_rank.map(([node, rank]) => (
              <li key={node}>Node {node} - Rank: {rank}</li>
            ))}
          </ol>
                </div>
            )}
        </>
    );
};

export default HITSPlot;
