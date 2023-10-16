import React, { useState, useEffect } from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
// import KMedoidsPlot from "./kmedoids_clusters.png"
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import ReactPaginate from 'react-paginate';

const K_Medoids_plot = ({ data, url, title }) => {

    const [cluster, setCluster] = useState([1,3,5,7,9]);
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

            } catch (error) {
                console.error('Error sending POST request:', error);
            }
        };
        fetchData();
        setLoading(false);
    }, [selectedValue]);

    return (
        <div class="d-flex flex-column">
          <FormControl fullWidth>
                <InputLabel id="cluster-label">Select Cluster</InputLabel>
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
                // !loading? <img src = {KMedoidsPlot} alt="Image is loading.." />:<h2>Processing.....</h2>
            }
           
        </div>
    );
};

export default K_Medoids_plot;
