import React, { useState, useEffect } from 'react';
import { Paper, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { Table } from 'antd';
import { DataGrid, GridToolbar } from "@mui/x-data-grid";
// import KMedoidsPlot from "./kmedoids_clusters.png"
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import ReactPaginate from 'react-paginate';

const PageRankPlot = ({ data, url, title }) => {

    const [fetchedData, setFetchedData] = useState([])
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

    return (
        <>
        {loading ? (
                <p>Loading...</p>
            ) : (
                <div>
                    <h1>PageRank </h1>
                    <TableContainer component={Paper}>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell>Page</TableCell>
                                    <TableCell>Rank</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {fetchedData && fetchedData.rank_table.map((row, index) => (
                                    <TableRow key={index}>
                                        <TableCell>{row.Page}</TableCell>
                                        <TableCell>{row.Rank}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </div>
            )}
        </>
    );
};

export default PageRankPlot;
