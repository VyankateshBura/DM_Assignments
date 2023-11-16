import React, { useState, useEffect } from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'

// import KMedoidsPlot from "./kmedoids_clusters.png"
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import ReactPaginate from 'react-paginate';

const DBSCANPLOT = ({ data, url, title }) => {

    const [cluster, setCluster] = useState([1,3,5,7,9]);
    const [clusters_scratch, setClusterScratch] = useState([]);
    const [clusters_builtin, setClusterBuiltin] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedValue, setSelectedValue] = useState(1); // Store the changed value of dropdown in selectedValue


    const handleDropdownChange = (e) => {
        setSelectedValue(e.target.value); // Set the selected value from the dropdown
    }

    useEffect(() => {
        setLoading(true);
        const fetchData = async () => {
            const requestData = {
                arrayData: data, // Use the selected value from the dropdown
                k: selectedValue
            };

            try {
                const response = await axios.post(`http://127.0.0.1:8000/api/v1/${url}/`, requestData);
                console.log("Response from backend ", response)
                console.log(JSON.parse(response.data.data['clusters_builtin'])['list2'])
                console.log(JSON.parse(response.data.data['clusters_scratch'])['list2'])
                setClusterScratch(JSON.parse(response.data.data['clusters_scratch'])['list2']);
                setClusterBuiltin(JSON.parse(response.data.data['clusters_builtin'])['list2']);

            } catch (error) {
                console.error('Error sending POST request:', error);
            }
        };
        fetchData();
        setLoading(false);
    }, [selectedValue]);

    console.log(clusters_scratch,clusters_builtin);

    return (
        <div class="d-flex flex-column">
          <FormControl fullWidth>
                <InputLabel id="cluster-label">Select Eps</InputLabel>
                <Select
                    labelId="cluster-label"
                    id="cluster"
                    value={selectedValue}
                    label="Select Cluster"
                    onChange={handleDropdownChange}
                >
                    {cluster.map((item, key) => (
                        <MenuItem value={item} key={key}>
                            {item}
                        </MenuItem>
                    ))}
                </Select>
                <InputLabel id="cluster-label">Select Minsamples</InputLabel>
                <Select
                    labelId="cluster-label"
                    id="cluster"
                    value={selectedValue}
                    label="Select Cluster"
                    onChange={handleDropdownChange}
                >
                    {cluster.map((item, key) => (
                        <MenuItem value={item} key={key}>
                            {item}
                        </MenuItem>
                    ))}
                </Select>
              
            </FormControl>
           
        {  
 
            clusters_scratch!=[] &&
            <TableContainer component={Paper}>
                            <Table aria-label="simple table">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Cluster from Scratch</TableCell>
                                        <TableCell>Data Points</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                { Object.keys(clusters_builtin).map((item, key) => (
                                                <TableRow key={key}>
                                                    <TableCell>{item}</TableCell>
                                                    {/* <TableCell>Data Points</TableCell> */}
                                                    <TableCell key={key}>
                                                    {clusters_builtin[item].join(', ')}
                                                     
                                                        </TableCell>
                                                        
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                }


                        {  
                        
                        clusters_scratch!=[] &&
                        <TableContainer component={Paper}>
                                        <Table aria-label="simple table">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Cluster from Builtin</TableCell>
                                                    <TableCell>Data Points</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                            { Object.keys(clusters_scratch).map((item, key) => (
                                                            <TableRow key={key}>
                                                                <TableCell>{item}</TableCell>
                                                                {/* <TableCell>Data Points</TableCell> */}
                                                                <TableCell key={key}>
                                                                {clusters_scratch[item].join(', ')}
                                                                
                                                                    </TableCell>
                                                                    
                                                                </TableRow>
                                                            ))}
                                                        </TableBody>
                                                    </Table>
                                                </TableContainer>
                        }
                     
                    
        </div>
    );
};

export default DBSCANPLOT;
