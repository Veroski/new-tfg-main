"""
Automated notebook testing with nbclient.

This module provides functionality to test generated notebooks
to ensure they execute without errors.
"""

import os
import sys
import logging
import nbformat
import tempfile
from nbclient import NotebookClient
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NotebookTester")

class NotebookTester:
    """
    Class for testing generated notebooks using nbclient.
    """
    
    def __init__(self, timeout: int = 600):
        """
        Initialize the notebook tester.
        
        Args:
            timeout: Maximum execution time for a notebook in seconds
        """
        self.timeout = timeout
    
    def test_notebook(self, notebook: nbformat.NotebookNode, skip_cells: List[int] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Test a notebook by executing it and capturing any errors.
        
        Args:
            notebook: The notebook to test
            skip_cells: List of cell indices to skip during execution
            
        Returns:
            Tuple containing success status and list of errors if any
        """
        # Create a temporary file for the notebook
        with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as temp_nb:
            temp_path = temp_nb.name
            nbformat.write(notebook, temp_nb)
        
        errors = []
        success = True
        
        try:
            # Configure the notebook client
            client = NotebookClient(
                notebook,
                timeout=self.timeout,
                kernel_name='python3',
                resources={'metadata': {'path': os.path.dirname(temp_path)}},
                skip_cells=skip_cells or []
            )
            
            # Execute the notebook
            logger.info(f"Executing notebook with {len(notebook.cells)} cells")
            client.execute()
            logger.info("Notebook execution completed successfully")
            
        except Exception as e:
            success = False
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': getattr(e, 'traceback', None)
            }
            errors.append(error_info)
            logger.error(f"Error executing notebook: {error_info['type']}: {error_info['message']}")
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
        
        return success, errors
    
    def test_notebook_cells(self, notebook: nbformat.NotebookNode) -> Dict[int, Dict[str, Any]]:
        """
        Test each cell of a notebook individually and report results.
        
        Args:
            notebook: The notebook to test
            
        Returns:
            Dictionary mapping cell indices to execution results
        """
        results = {}
        
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
                
            # Create a mini-notebook with just this cell and any imports/setup
            mini_notebook = nbformat.v4.new_notebook()
            
            # Add any setup cells (typically the first few cells with imports)
            setup_cells = [0, 1, 2]  # Usually the first 3 cells are setup
            for setup_idx in setup_cells:
                if setup_idx < len(notebook.cells) and notebook.cells[setup_idx].cell_type == 'code':
                    mini_notebook.cells.append(notebook.cells[setup_idx])
            
            # Add the cell to test if it's not already included
            if i not in setup_cells:
                mini_notebook.cells.append(cell)
            
            # Test the mini-notebook
            success, errors = self.test_notebook(mini_notebook)
            
            results[i] = {
                'success': success,
                'errors': errors
            }
            
            if success:
                logger.info(f"Cell {i} executed successfully")
            else:
                logger.error(f"Cell {i} execution failed: {errors}")
        
        return results
    
    def validate_notebook_structure(self, notebook: nbformat.NotebookNode) -> Tuple[bool, List[str]]:
        """
        Validate the structure of a notebook without executing it.
        
        Args:
            notebook: The notebook to validate
            
        Returns:
            Tuple containing validity status and list of issues if any
        """
        issues = []
        
        # Check for basic notebook structure
        if not hasattr(notebook, 'cells'):
            issues.append("Notebook has no cells attribute")
            return False, issues
        
        if not notebook.cells:
            issues.append("Notebook has no cells")
            return False, issues
        
        # Check for metadata
        if not hasattr(notebook, 'metadata'):
            issues.append("Notebook has no metadata attribute")
        
        # Check for code cells
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        if not code_cells:
            issues.append("Notebook has no code cells")
        
        # Check for imports in the first few cells
        import_found = False
        for i, cell in enumerate(notebook.cells[:3]):
            if cell.cell_type == 'code' and ('import ' in cell.source or 'from ' in cell.source):
                import_found = True
                break
        
        if not import_found:
            issues.append("No import statements found in the first few cells")
        
        # Check for inference function
        inference_fn_found = False
        for cell in notebook.cells:
            if cell.cell_type == 'code' and 'def inference_fn' in cell.source:
                inference_fn_found = True
                break
        
        if not inference_fn_found:
            issues.append("No inference_fn function defined in the notebook")
        
        return len(issues) == 0, issues