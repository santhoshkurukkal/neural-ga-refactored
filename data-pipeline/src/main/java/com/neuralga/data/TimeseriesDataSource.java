package com.neuralga.data;

import java.nio.file.Path;
import java.util.List;

/**
 * Interface for time-series data sources.
 * Implementations can read from files, databases, or generate synthetic data.
 */
public interface TimeseriesDataSource {

    /**
     * Load the time-series data.
     * @return List of values in temporal order
     */
    List<Double> load();

    /**
     * Load data from a specific path.
     * @param path Path to data file
     * @return List of values
     */
    List<Double> loadFromPath(Path path);

    /**
     * Get the name/identifier of this data source.
     * @return Source name
     */
    String getSourceName();
}