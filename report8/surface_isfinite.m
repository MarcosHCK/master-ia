function result = surface_isfinite(values)
%SURFACE_ISFINITE MATLAB/Octave-compatible finite-value check.
result = ~isnan(values) & ~isinf(values);
end