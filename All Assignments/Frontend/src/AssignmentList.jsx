import React from 'react';
import { Link } from 'react-router-dom';

const assignments = [
    {
      id: 1,
      title: 'Assignment 1',
      sections: [
        { id: 1, title: 'Dashboard', content: 'Content of section 1...' },
        { id: 2, title: 'Central Tendency', content: 'Content of section 2...' },
        { id: 3, title: 'Dispersion', content: 'Content of section 3...' },
        { id: 4, title: 'Histogram', content: 'Content of section 4...' },
        { id: 5, title: 'ScatterPlot', content: 'Content of section 5...' },
        { id: 6, title: 'BoxPlot', content: 'Content of section 6...' },
        // ... More sections for Assignment 1
      ],
    },
    {
      id: 2,
      title: 'Assignment 2',
      sections: [
        { id: 1, title: 'Section 1', content: 'Content of section 1...' },
        { id: 2, title: 'Section 2', content: 'Content of section 2...' },
        // ... More sections for Assignment 2
      ],
    },
    // ... More assignments
  ];
  

function AssignmentList() {
  return (
    <div>
      <h2>Assignment List</h2>
      <ul>
        {assignments.map(assignment => (
          <li key={assignment.id}>
            <Link to={`/assignments/${assignment.id}`}>{assignment.title}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default AssignmentList;
