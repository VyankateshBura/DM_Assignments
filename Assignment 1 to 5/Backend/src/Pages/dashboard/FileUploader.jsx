import React, { useState ,useContext} from 'react';
import { Button } from '@mui/material';
import { CircularProgress,Typography } from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
// import {setData} from "../Data"
import { FileData } from '../../App';

const FileUploader = () => {
  const conn = useContext(FileData)
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState(null);
  const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setUploadProgress(0); // Reset progress
        setUploadError(null); // Reset error message
        // Read the content of the file using FileReader
        const reader = new FileReader();

        reader.onload = (event) => {
        const content = event.target.result;
        setFileContent(content);
        // Parse the content based on file type (CSV or Excel)
        if (file.name.endsWith('.csv')) {
            parseCSVContent(content);
        } else if (file.name.endsWith('.xlsx')) {
            parseExcelContent(content);
        }
        };
        reader.onprogress = (event) => {
            if (event.lengthComputable) {
              const progress = (event.loaded / event.total) * 100;
              setUploadProgress(progress);
            }
          };
        reader.onerror = () => {
            setUploadError('Error occurred while processing the file.');
        };

        reader.readAsBinaryString(file);
  };

  const parseCSVContent = (content) => {
    const result = Papa.parse(content, {
      header: true, // Treat the first row as headers
      skipEmptyLines: true,
    });
    conn.setFdata(result.data);
    console.log('CSV Content:', result.data);
  };

  const parseExcelContent = (content) => {
    const workbook = XLSX.read(content, { type: 'binary' });
    const sheetName = workbook.SheetNames[0]; // Assuming we read the first sheet
    const worksheet = workbook.Sheets[sheetName];
    const data = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
    conn.setFdata(data);
    console.log('Excel Content:', data);
  };

  return (
    <div style={{ textAlign: 'center' }}>
        {uploadError && (
        <Typography variant="body1" color="error" style={{ marginTop: '10px' }}>
          {uploadError}
        </Typography>
      )}
      <input
        type="file"
        accept=".csv, .xlsx"
        style={{ display: 'none' }}
        id="file-input"
        onChange={handleFileChange}
      />
      <label htmlFor="file-input">
        <Button
          variant="contained"
          component="span"
          startIcon={<CloudUploadIcon />}
          color="primary"
        >
          Upload CSV/Excel File
        </Button>
      </label>

      {uploadProgress > 0 && uploadProgress < 100 && (
        <div style={{ marginTop: '20px' }}>
          <CircularProgress variant="determinate" value={uploadProgress} />
          <div>{Math.round(uploadProgress)}% Uploaded</div>
        </div>
      )}

    {fileContent && !uploadError && uploadProgress === 100 && (
            <Typography variant="body1" color="success" style={{ marginTop: '10px' }}>
            File Successfully Uploaded
            </Typography>
        )}
    </div>
  );
};

export default FileUploader;
