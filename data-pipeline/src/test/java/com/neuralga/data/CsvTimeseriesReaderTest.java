package com.neuralga.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class CsvTimeseriesReaderTest {

    @Test
    void shouldReadSimpleCsvFile(@io.TempDir Path tempDir) {
        Path file = tempDir.resolve("test.csv");
        java.nio.file.Files.writeString(file, "1.0\n2.0\n3.0\n4.0\n5.0\n");

        CsvTimeseriesReader reader = new CsvTimeseriesReader();
        List<Double> data = reader.loadFromPath(file);

        assertThat(data).containsExactly(1.0, 2.0, 3.0, 4.0, 5.0);
    }

    @Test
    void shouldSkipCommentsAndEmptyLines(@io.TempDir Path tempDir) {
        Path file = tempDir.resolve("test.csv");
        java.nio.file.Files.writeString(file, "# Comment\n1.0\n\n2.0\n# Another\n3.0\n");

        CsvTimeseriesReader reader = new CsvTimeseriesReader();
        List<Double> data = reader.loadFromPath(file);

        assertThat(data).containsExactly(1.0, 2.0, 3.0);
    }

    @Test
    void shouldReadCsvWithHeader(@io.TempDir Path tempDir) {
        Path file = tempDir.resolve("test.csv");
        java.nio.file.Files.writeString(file, "value,timestamp\n1.0,1\n2.0,2\n3.0,3\n");

        CsvTimeseriesReader reader = new CsvTimeseriesReader(true, ",", 0);
        List<Double> data = reader.loadFromPath(file);

        assertThat(data).containsExactly(1.0, 2.0, 3.0);
    }

    @Test
    void shouldReadMultiColumnCsv(@io.TempDir Path tempDir) {
        Path file = tempDir.resolve("test.csv");
        java.nio.file.Files.writeString(file, "timestamp,value,label\n1,10.5,A\n2,20.3,B\n3,30.1,C\n");

        CsvTimeseriesReader reader = new CsvTimeseriesReader(true, ",", 1);
        List<Double> data = reader.loadFromPath(file);

        assertThat(data).containsExactly(10.5, 20.3, 30.1);
    }

    @Test
    void shouldThrowOnEmptyFile(@io.TempDir Path tempDir) {
        Path file = tempDir.resolve("empty.csv");
        java.nio.file.Files.writeString(file, "");

        CsvTimeseriesReader reader = new CsvTimeseriesReader();
        assertThatThrownBy(() -> reader.loadFromPath(file))
                .isInstanceOf(CsvTimeseriesReader.DataLoadingException.class)
                .hasMessageContaining("No valid data found");
    }

    @Test
    void shouldThrowOnNonExistentFile() {
        CsvTimeseriesReader reader = new CsvTimeseriesReader();
        assertThatThrownBy(() -> reader.loadFromPath(Path.of("nonexistent.csv")))
                .isInstanceOf(CsvTimeseriesReader.DataLoadingException.class);
    }
}