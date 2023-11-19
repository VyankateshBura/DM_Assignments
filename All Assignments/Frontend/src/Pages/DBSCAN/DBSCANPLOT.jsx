import React, { useState, useEffect } from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, Button ,TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import DBSCANIMG from "./DBSCAN.png"
// import KMedoidsPlot from "./kmedoids_clusters.png"
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import ReactPaginate from 'react-paginate';

const DBSCANPLOT = ({ data, url, title }) => {

    const [clusters, setClusters] = useState([]);
    const [min_samples, setmin_samples] = useState([1,3,5,7,9]);
    const [eps, seteps] = useState([0.1,0.3,0.5,0.7,0.9]);
    const [clusters_scratch, setClusterScratch] = useState([]);
    const [clusters_builtin, setClusterBuiltin] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedValue, setSelectedValue] = useState(1); // Store the changed value of dropdown in selectedValue
    const [epsValue, setEpsValue] = useState(1);
    const [minSamplesValue, setMinSamplesValue] = useState(1);

    const handleEpsChange = (e) => {
        setEpsValue(e.target.value);
    };

    const handleMinSamplesChange = (e) => {
        setMinSamplesValue(e.target.value);
    };


    const handleDropdownChange = (e) => {
        setSelectedValue(e.target.value); // Set the selected value from the dropdown
    }
    const fetchData = async () => {
        const requestData = {
            arrayData: data, // Use the selected value from the dropdown
            eps: epsValue,
            min_samples: minSamplesValue,
        };

        try {
            const response = await axios.post(`http://127.0.0.1:8000/api/v1/${url}/`, requestData);
            console.log("Response from backend ", response)
            console.log(JSON.parse(response.data.data))
            let responseFromBackend = JSON.parse(response.data.data)
            const { colors, unique_labels } = responseFromBackend;

            // Update state with the new data structure
            setClusters(colors.map((color, index) => ({ label: unique_labels[index], color })));

            // console.log(JSON.parse(response.data.data['clusters_scratch'])['list2'])
            // setClusterScratch(JSON.parse(response.data.data['clusters_scratch'])['list2']);
            // setClusterBuiltin(JSON.parse(response.data.data['clusters_builtin'])['list2']);

        } catch (error) {
            console.error('Error sending POST request:', error);
        }
    };


    console.log(clusters_scratch,clusters_builtin);

    return (
        <div class="d-flex flex-column">
          <FormControl fullWidth>
    <InputLabel id="eps-label">Select Eps</InputLabel>
    <Select
        labelId="eps-label"
        id="eps"
        value={epsValue}
        label="Select Eps"
        onChange={handleEpsChange}
    >
        {eps.map((item, key) => (
            <MenuItem value={item} key={key}>
                {item}
            </MenuItem>
        ))}
    </Select>
    <InputLabel id="min-samples-label">Select Min Samples</InputLabel>
    <Select
        labelId="min-samples-label"
        id="min-samples"
        value={minSamplesValue}
        label="Select Min Samples"
        onChange={handleMinSamplesChange}
    >
        {min_samples.map((item, key) => (
            <MenuItem value={item} key={key}>
                {item}
            </MenuItem>
        ))}
    </Select>
    {/* Add a button to trigger the backend request */}
    <Button variant="contained" color="primary" onClick={fetchData}>
        Submit
    </Button>
</FormControl>
           <img src = {DBSCANIMG}/>
            {clusters.length > 0 && (
                <TableContainer component={Paper}>
                    <Table aria-label="simple table">
                        <TableHead>
                            <TableRow>
                                <TableCell>Cluster Label</TableCell>
                                <TableCell>Data Points</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {clusters.map((cluster, key) => (
                                <TableRow key={key}>
                                    <TableCell>{cluster.label}</TableCell>
                                    <TableCell>
                                        {cluster.color.join(', ')}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            )}
       
                     
                    
        </div>
    );
};

export default DBSCANPLOT;
