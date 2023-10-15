import React, { useState, useEffect ,useContext} from 'react';
import Drawer from '@material-ui/core/Drawer';
import { useTheme } from "@mui/material";
// import { tokens } from "../theme";
import { DataGrid, GridToolbar } from "@mui/x-data-grid";
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import { ThemeProvider } from "@material-ui/core/styles"
import ColorPicker  from 'material-ui-color-picker';
import HighlightIcon from '@mui/icons-material/Highlight';
import HideImageIcon from '@mui/icons-material/HideImage';
import RemoveCircleIcon from '@mui/icons-material/RemoveCircle';
import ViewStreamIcon from '@mui/icons-material/ViewStream';
import { FileData } from '../../App';
import "./logtable.css"
import { Navigate, useNavigate } from 'react-router-dom';


const LogsTable = ({ logs,totallogs }) => {
  const theme = useTheme();
  const conn = useContext(FileData)
  // const colors = tokens(theme.palette.mode);
  const filteredRows = conn.fdata!=null?conn.fdata.map((row, index) => ({ ...row, id: index + 1 })):[];
  const columns = conn.fdata!=null?Object.keys(conn.fdata[0]).map((key) => ({
    field: key,
    headerName: key,
    width: 150,
  })):[];
  return (

      <ThemeProvider theme={theme}>
        <DataGrid
          rows={filteredRows}
          columns={columns}
          components={{ Toolbar: GridToolbar }}
          initialState={{
            pagination:{
            paginationModel: { pageSize: 5},
          }}}
        />
      </ThemeProvider>
  );
};

export default LogsTable;
