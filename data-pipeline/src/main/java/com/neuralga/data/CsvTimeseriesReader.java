package com.neuralga.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class CsvTimeseriesReader implements TimeseriesDataSource {

    private static final Logger log = LoggerFactory.getLogger(CsvTimeseriesReader.class);

    private final boolean hasHeader;
    private final String delimiter;
    private final int valueColumnIndex;

    public CsvTimeseriesReader() {
        this(false, ",", 0);
    }

    public CsvTimeseriesReader(boolean hasHeader, String delimiter, int valueColumnIndex) {
        this.hasHeader = hasHeader;
        this.delimiter = delimiter;
        this.valueColumnIndex = valueColumnIndex;
    }

    @Override
    public List<Double> load() {
        throw new UnsupportedOperationException("Use loadFromPath(Path) instead");
    }

    @Override
    public List<Double> loadFromPath(Path path) {
        List<Double> values = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String line;
            boolean firstLine = true;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }

                if (firstLine && hasHeader) {
                    firstLine = false;
                    continue;
                }

                String[] parts = line.split(delimiter);
                if (parts.length > valueColumnIndex) {
                    try {
                        double value = Double.parseDouble(parts[valueColumnIndex].trim());
                        values.add(value);
                    } catch (NumberFormatException e) {
                        log.warn("Skipping invalid number at line: {}", line);
                    }
                }
            }

            log.info("Loaded {} values from {}", values.size(), path);
        } catch (IOException e) {
            log.error("Failed to read data from: {}", path, e);
            throw new DataLoadingException("Failed to read data from " + path, e);
        }

        if (values.isEmpty()) {
            throw new DataLoadingException("No valid data found in: " + path);
        }

        return values;
    }

    @Override
    public String getSourceName() {
        return "CSV Time-Series Reader";
    }

    public static class DataLoadingException extends RuntimeException {
        public DataLoadingException(String message, Throwable cause) {
            super(message, cause);
        }

        public DataLoadingException(String message) {
            super(message);
        }
    }
}