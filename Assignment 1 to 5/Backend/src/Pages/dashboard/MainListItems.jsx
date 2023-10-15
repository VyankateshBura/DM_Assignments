import * as React from 'react';
import {useState} from 'react'
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import ListSubheader from '@mui/material/ListSubheader';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import PeopleIcon from '@mui/icons-material/People';
import BarChartIcon from '@mui/icons-material/BarChart';
import LayersIcon from '@mui/icons-material/Layers';
import List from '@mui/material/List';
import FunctionsIcon from '@mui/icons-material/Functions';
import ScatterPlotIcon from '@mui/icons-material/ScatterPlot';
import AssignmentIcon from '@mui/icons-material/Assignment';
import BallotIcon from '@mui/icons-material/Ballot';
import {Link} from 'react-router-dom'
const MainListItems = ()=>{
    const [expandedAssignment, setExpandedAssignment] = useState(null);
    const assignments = [
      {
        id: 1,
        title: 'Assignment 1',
        sections: [
          { id: 1, title: 'File/CSV Upload', content: 'Content of section 1...' },
          { id: 2, title: 'Central Tendency', content: 'Content of section 2...' },
          { id: 3, title: 'Histogram', content: 'Content of section 3...' },
          { id: 4, title: 'Dispersion', content: 'Content of section 4...' },
          { id: 5, title: 'ScatterPlot', content: 'Content of section 5...' },
          { id: 6, title: 'BoxPlot', content: 'Content of section 6...' },
          { id: 7, title: 'Quantile-Quantile Plot', content: 'Content of section 7...' },
          // ... More sections for Assignment 1
        ],
      },
      {
        id: 2,
        title: 'Assignment 2',
        sections: [
          { id: 1, title: 'File/CSV Upload', content: 'Content of section 1...' },
          { id: 2, title: 'Chi Square Test', content: 'Content of section 2...' },
          { id: 3, title: 'Min Max', content: 'Content of section 3...' },
          { id: 4, title: 'Decimal Scaling', content: 'Content of section 4...' },
          { id: 5, title: 'Z-Score', content: 'Content of section 5...' },
          { id: 6, title: 'Correlation (Pearson Coefficient)', content: 'Content of section 6...' },
          // ... More sections for Assignment 2
        ],
      },
      {
        id: 3,
        title: 'Assignment 3',
        sections: [
          { id: 1, title: 'File/CSV Upload', content: 'Content of section 1...' },
          { id: 2, title: 'DecisionTree', content: 'Content of section 2...' },
        ],
      },
      {
        id: 4,
        title: 'Assignment 4',
        sections: [
          { id: 1, title: 'File/CSV Upload', content: 'Content of section 1...' },
          { id: 2, title: 'Rule Based Evaluation', content: 'Content of section 2...' },
        ],
      },
      {
        id: 5,
        title: 'Assignment 5',
        sections: [
          { id: 1, title: 'File/CSV Upload', content: 'Content of section 1...' },
          { id: 2, title: ' Classifiers', content: 'Content of section 2...' },
          // { id: 3, title: 'Naive Bayesian', content: 'Content of section 3...' },
          // { id: 4, title: 'KNN', content: 'Content of section 4...' },
          // { id: 5, title: 'ANN', content: 'Content of section 5...' },
          
        ],
      },
      // ... More assignments
    ];
    const handleAssignmentClick = (assignment) => {
      if (expandedAssignment === assignment) {
        setExpandedAssignment(null);
      } else {
        setExpandedAssignment(assignment);
      }
    };
    return (
      <React.Fragment>
      
     <ListSubheader>Assignments</ListSubheader>
          <List>

            {
              assignments.map((item,key)=>{
                return(
                  <>
                  <div key={item.id}>
                  <ListItemButton onClick={() => handleAssignmentClick(item.title)} >
                  <ListItemIcon>
                    <AssignmentIcon />
                  </ListItemIcon>
                  <ListItemText primary={item.title} />
                </ListItemButton>
                {expandedAssignment === item.title && (
                  <List>
                    {
                      item.sections.map((Subitem,subkey)=>{
                          return(
                            <ListItemButton>
                            {/* Icon for Section 1 of Assignment 1 */}
                            &nbsp;&nbsp;&nbsp;
                            <BallotIcon/>
                            &nbsp;&nbsp;&nbsp;
                            <Link to={`/assignment${item.id}/section${Subitem.id}`} className="link" key={Subitem.id}>
                              <ListItemText primary={Subitem.title} />
                            </Link>
                          </ListItemButton>
                          )
                      })
                    }
                   
                    {/* Add more sections for Assignment 1 */}
                  </List>
                )}
                  </div>
          
                
                </>
                )
              })
            }

            
            
          </List>
    </React.Fragment>
    )
  }

  export default MainListItems;