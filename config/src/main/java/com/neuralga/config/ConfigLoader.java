package com.neuralga.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ConfigLoader {

    private static final Logger log = LoggerFactory.getLogger(ConfigLoader.class);
    private static final ObjectMapper YAML_MAPPER = new ObjectMapper(new YAMLFactory());

    static {
        YAML_MAPPER.findAndRegisterModules();
    }

    private ConfigLoader() {}

    public static TrainingConfig loadTrainingConfig(String configPath) {
        return loadConfig(configPath, TrainingConfig.class, "training");
    }

    public static GAConfig loadGAConfig(String configPath) {
        return loadConfig(configPath, GAConfig.class, "ga");
    }

    public static CascadeConfig loadCascadeConfig(String configPath) {
        return loadConfig(configPath, CascadeConfig.class, "cascade");
    }

    public static AppConfig loadAppConfig(String configPath) {
        return loadConfig(configPath, AppConfig.class, "app");
    }

    public static <T> T loadConfig(String configPath, Class<T> configClass, String configName) {
        Path path = Paths.get(configPath);
        if (!Files.exists(path)) {
            log.warn("Config file not found: {}, using defaults for {}", configPath, configName);
            return createDefault(configClass);
        }

        try (InputStream is = Files.newInputStream(path)) {
            T config = YAML_MAPPER.readValue(is, configClass);
            log.info("Loaded {} config from: {}", configName, configPath);
            return config;
        } catch (IOException e) {
            log.error("Failed to load {} config from: {}", configName, configPath, e);
            return createDefault(configClass);
        }
    }

    public static <T> T loadConfigFromResource(String resourcePath, Class<T> configClass) {
        try (InputStream is = ConfigLoader.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (is == null) {
                log.warn("Resource not found: {}, using defaults", resourcePath);
                return createDefault(configClass);
            }
            T config = YAML_MAPPER.readValue(is, configClass);
            log.info("Loaded config from resource: {}", resourcePath);
            return config;
        } catch (IOException e) {
            log.error("Failed to load config from resource: {}", resourcePath, e);
            return createDefault(configClass);
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> T createDefault(Class<T> configClass) {
        try {
            return configClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            log.error("Failed to create default config for: {}", configClass.getName(), e);
            throw new IllegalStateException("Cannot create default config", e);
        }
    }

    public static void saveConfig(Object config, String outputPath) {
        Path path = Paths.get(outputPath);
        try {
            Files.createDirectories(path.getParent());
            YAML_MAPPER.writerWithDefaultPrettyPrinter().writeValue(Files.newOutputStream(path), config);
            log.info("Saved config to: {}", outputPath);
        } catch (IOException e) {
            log.error("Failed to save config to: {}", outputPath, e);
            throw new IllegalStateException("Cannot save config", e);
        }
    }
}