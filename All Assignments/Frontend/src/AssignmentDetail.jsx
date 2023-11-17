import React from 'react';
import { Link } from 'react-router-dom';

function AssignmentDetails({ match ,assignments}) {
  const { assignmentId } = match.params;

  // Fetch assignment details using assignmentId
  const assignment = assignments.find(a => a.id === parseInt(assignmentId));

  if (!assignment) {
    return <div>Assignment not found</div>;
  }

  return (
    <div>
      <h2>Assignment Details: {assignment.title}</h2>
      <ul>
        {assignment.sections.map(section => (
          <li key={section.id}>
            <Link to={`/assignments/${assignmentId}/section/${section.id}`}>
              {section.title}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default AssignmentDetails;
