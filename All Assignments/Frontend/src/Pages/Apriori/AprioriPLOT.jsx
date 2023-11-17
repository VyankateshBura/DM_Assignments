import React, { useState, useEffect } from 'react';
import { Paper, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { Table } from 'antd';
import { DataGrid, GridToolbar } from "@mui/x-data-grid";
// import KMedoidsPlot from "./kmedoids_clusters.png"
import { FormControl, InputLabel, MenuItem, Select } from '@mui/material';
import ReactPaginate from 'react-paginate';

const AprioriPLOT = ({ data, url, title }) => {

    const [fetchedData, setFetchedData] = useState([])
    const [loading, setLoading] = useState(false);
   

    const columns = [
        {
          field: 'Antecedents',
          headerName: 'antecedents',
          flex:1,
          width: 250,
          // key: 'antecedents',
          // render: antecedents => antecedents.toString()
        },
        {
          field: 'Consequents',
          headerName: 'consequents',
          // key: 'consequents',
          flex:1
          // render: consequents => consequents.toString()
        },
        // {
        //   field: 'Support',
        //   headerName: 'support',
        //   key: 'support'
        // },
        {
          field: 'Confidence',
          headerName: 'confidence',
          // key: 'confidence'
          flex:1
        },
        {
          field: 'Lift',
          headerName: 'lift',
          // key: 'lift',
          flex:1
        },
        // {
        //   field: 'Leverage',
        //   headerName: 'leverage',
        //   key: 'leverage'
        // },
        // {
        //   field: 'Conviction',
        //   headerName: 'conviction',
        //   key: 'conviction'
        // },
        // {
        //   field: "Zhang's Metric",
        //   headerName: 'zhangs_metric',
        //   key: 'zhangs_metric'
        // },
        // {
        //   field: 'Kulczynski',
        //   headerName: 'kulczynski',
        //   key: 'kulczynski'
        // }
      ];
  



    useEffect(() => {
        setLoading(true);
        const fetchData = async () => {
            const requestData = {
                arrayData: data, // Use the selected value from the dropdown
               
            };
            
            try {
                const response = await axios.post(`http://127.0.0.1:8000/api/v1/${url}/`, requestData);
               
                console.log("Response from backend ", response)
                
                // let cleanedData = response.data.replace(/Infinity/g, '"Infinity"')
                // let parsedData = JSON.parse(cleanedData)
  
                let dt = response.data.results[0].Rules
                let n = response.data.results[0].total_rules
                let finalTemp = []
                for(let i=0;i<n;i++){
                  let temp = {}
                  for(let j=0;j<columns.length;j++){
                      if(j<2){
                        // console.log(columns[j].field)
                        temp[columns[j].field]=dt[i][j].toString()
                      
                      }
                      else{
                        // console.log(dt[i][j])
                        temp[columns[j].field]=dt[i][j]
                      }
                  }
                  temp['id']= finalTemp.length+1
                  finalTemp.push(temp)
                  
                }

                // console.log(finalTemp)
                setFetchedData(finalTemp)
                
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
                    <h1>Apriori Table</h1>
      
                    <DataGrid
                      rows={fetchedData}
                      columns={columns}
                      components={{ Toolbar: GridToolbar }}
                      initialState={{
                        pagination:{
                        paginationModel: { pageSize: 5},
                      }}}
                    />
                </div>
            )}
        </>
    );
};

export default AprioriPLOT;
