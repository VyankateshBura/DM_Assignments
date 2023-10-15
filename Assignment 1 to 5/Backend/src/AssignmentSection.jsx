import React from 'react';
import { useParams } from 'react-router-dom';

function AssignmentSection() {
  const { assignmentId, sectionId } = useParams();

  // Fetch section content using assignmentId and sectionId

  return (
    <div>
      <h2>Assignment Section {sectionId}</h2>
      {/* Display section content */}
    </div>
  );
}

export default AssignmentSection;
