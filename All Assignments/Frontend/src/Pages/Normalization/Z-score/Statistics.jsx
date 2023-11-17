import React, { useState,useEffect } from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios'
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
import ScatterPlotChart from '../../ScatterPlot/ScatterPlotChart';

import ReactPaginate from 'react-paginate';

const Statistics = ({ data ,url,title}) => {
  const [zscoreData, setZscoreData] = useState([]);
  const [zscoreEstimatedData, setZscoreEstimatedData] = useState([]);
  const itemsPerPage = 10;
  const [currentPage, setCurrentPage] = useState(0);

  const handlePageChange = (selectedPage) => {
    setCurrentPage(selectedPage.selected);
  };
  useEffect(() => {
    const fetchData = async () => {
      const requestData = {
        arrayData: data,
        var1: "Hi",
        var2: "Hello"
      };

      try {
        const response = await axios.post(`http://127.0.0.1:8000/api/v1/${url}/`, requestData);
        console.log("Response from backend ",response.data.data[0].data)
        setZscoreData(response.data.data[0].data);
        setZscoreEstimatedData(response.data.data[1].data);
      } catch (error) {
        console.error('Error sending POST request:', error);
      }
    };

    if (data.length > 0) {
      fetchData();
    }
  }, [data]);

  
  const columnNames = Object.keys(data[0]);

  const getColumnData = (columnName) => data.map((row) => row[columnName]);

  const calculateMean = (columnName) => {
    const columnData = getColumnData(columnName);
    const sum = columnData.reduce((acc, val) => acc + parseFloat(val), 0);
    return sum / columnData.length;
  };

  
 



  const renderTable = (dataArray) => {
    const getRowId = (row) => row.id || row._id || dataArray.data.indexOf(row);
    const columns = dataArray.columns?dataArray.columns.map((column,key)=>({
      field:column, headerName: column, width: 100 
    })):[]
  
    // Convert the array of arrays into an array of objects with named properties
    const rows  = dataArray.data?dataArray.data.map((rowArray, index) => {
      const row = { id: index };
      columns.forEach((column, columnIndex) => {
        row[column.field] = rowArray[columnIndex];
      });
      return row;
    }):[];
    // const [rows, setRows] = useState(dataArray.data.slice(0, 10));
 
    
     // Show only 10 rows initially

    const handlePageChange = (params) => {
      const { page, pageSize } = params;
      const startIndex = page * pageSize;
      const endIndex = startIndex + pageSize;
      setRows(rows.slice(startIndex, endIndex));
    };
    const loadMoreRows = () => {
      const currentRowCount = rows.length;
      const nextRows = dataArray.slice(currentRowCount, currentRowCount + 10);
      setRows((prevRows) => [...prevRows, ...nextRows]);
    };
  

    return (
      <div>
      {dataArray.data&&dataArray.columns&&<DataGrid
          rows={rows}
          columns={columns}
          components={{ Toolbar: GridToolbar }}
          density={'compact'}
          initialState={{
            pagination:{
            paginationModel: { pageSize: 5},
          }}}
          maxPageSize={25}
          loading={rows.length === 0} // Set loading to true when no rows are loaded
          onRowsScrollEnd={loadMoreRows} // Load more rows when scrolling reaches the end
          // getRowId={getRowId}
        />}
  </div>
    )
    };
  return (
    <div>
          <div>
          <div>
            <h2>{title} Data</h2>
            {zscoreData.length!=[] ?renderTable(zscoreData):""}
          </div>

          <div>
            <h2>{title} Estimated Data</h2>
            {zscoreData.length!=[] ?renderTable(zscoreEstimatedData):""}
          </div>
        </div>

      {/* Additional visualizations */}
      <div style={{ marginTop: '40px' }}>
        {/* {columnNames.map((columnName) => (
          <div key={columnName} style={{ marginBottom: '20px' }}>
            <h3>{columnName}</h3>
            <BarChart width={800} height={300} data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={columnName} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey={columnName} fill="#8884d8" name={columnName} />
            </BarChart>
          </div>
        ))} */}
        <h4>ScatterPlot of the {title} Normalisation by traditional Method</h4>
        {zscoreEstimatedData.length!=[] ?<ScatterPlotChart data= {zscoreEstimatedData.data}/>:""}

        <h4>ScatterPlot of the {title} Normalisation by Built In Method</h4>
        {zscoreData.length!=[] ?<ScatterPlotChart data= {zscoreData.data}/>:""}
      </div>
    </div>
  );
};

export default Statistics;
