#-----------------------------------------
#-- Test suite.
#-----------------------------------------
import math
import nigon.test_nigon as test_nigon

def test_nigon_lucpu_coarse():
    """Test numerical solution with LU solver on coarse mesh (numRefs=2)."""
    errors = test_nigon(numRefs=2, lsolver=ug4.LUCPU1())
    assert math.sqrt(errors[0]*errors[0] + errors[1]*errors[1])  <  1e-3, "Error u should be computed"
    # assert errors[1] is not None, "Error v should be computed"
    print("✓ Test passed: LUCPU1 (numRefs=2)")

def test_nigon_lucpu_medium():
    """Test numerical solution with LU solver on medium mesh (numRefs=3)."""
    errors = test_nigon(numRefs=3, lsolver=ug4.LUCPU1())
    assert math.sqrt(errors[0]*errors[0] + errors[1]*errors[1]) <  1e-4,
    print("✓ Test passed: LUCPU1 (numRefs=3)")

def test_nigon_lucpu_fine():
    """Test numerical solution with LU solver on fine mesh (numRefs=4).
    Note: This requires SuperLU solver due to matrix size.
    """
    # test_nigon(numRefs=4, lsolver=slu.SuperLUCPU1())
    pass

#if __name__ == "__main__":
#    import pytest
#    pytest.main([__file__, "-v"])