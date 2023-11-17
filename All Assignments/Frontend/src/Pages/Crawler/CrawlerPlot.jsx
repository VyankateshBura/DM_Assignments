import React, { useState, useEffect } from 'react';
import { Paper, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { Table } from 'antd';
import { DataGrid, GridToolbar } from "@mui/x-data-grid";
// import KMedoidsPlot from "./kmedoids_clusters.png"
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import ReactPaginate from 'react-paginate';

const CrawlerPlot = ({ data, url, title }) => {


    const [loading, setLoading] = useState(false);
    const [seedUrl, setSeedUrl] = useState('');
    const [links, setLinks] = useState([]);



    const fetchData = async () => {
        const requestData = {
            seed_url: seedUrl // Use the selected value from the dropdown
           
        };
        
        try {
            const response = await axios.post(`http://127.0.0.1:8000/api/v1/crawl/`, requestData);
           
            console.log("Response from backend ", response)
            
            setLinks(response.data.links)
            
        } catch (error) {
            console.error('Error sending POST request:', error);
        }
    };
    


    // Function to extract row id

    return (
        <>
        {loading ? (
                <p>Loading...</p>
            ) : (
                <div>
                    <input type="text" value={seedUrl} onChange={(e) => setSeedUrl(e.target.value)} />
                    <button onClick={fetchData}>Crawl</button>
                    <br></br>
                    <br></br>
                    <br></br>
                    {links.length?<h3>Links Obtained :</h3>:""}
                    <ul>
                        {links.map((link, index) => (
                            <li key={index}>
                                <a href={link} target="_blank" rel="noopener noreferrer">
                                    {link}
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </>
    );
};

export default CrawlerPlot;
