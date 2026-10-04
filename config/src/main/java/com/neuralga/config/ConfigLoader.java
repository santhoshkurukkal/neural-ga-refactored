package com.neuralga.config;

import com.fasterxml.jackson.databind.JsonNode;
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
        YAML_MAPPER.configure(com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        YAML_MAPPER.setPropertyNamingStrategy(com.fasterxml.jackson.databind.PropertyNamingStrategies.SNAKE_CASE);
    }

    private ConfigLoader() {}

    public static TrainingConfig loadTrainingConfig(String configPath) {
        JsonNode root = loadYamlRoot(configPath);
        return loadSection(root, "training", TrainingConfig.class);
    }

    public static GAConfig loadGAConfig(String configPath) {
        JsonNode root = loadYamlRoot(configPath);
        return loadSection(root, "ga", GAConfig.class);
    }

    public static CascadeConfig loadCascadeConfig(String configPath) {
        JsonNode root = loadYamlRoot(configPath);
        return loadSection(root, "cascade", CascadeConfig.class);
    }

    public static AppConfig loadAppConfig(String configPath) {
        JsonNode root = loadYamlRoot(configPath);
        return loadSection(root, "app", AppConfig.class);
    }

    public static <T> T loadConfig(String configPath, Class<T> configClass, String configName) {
        JsonNode root = loadYamlRoot(configPath);
        return loadSection(root, configName, configClass);
    }

    private static JsonNode loadYamlRoot(String configPath) {
        Path path = Paths.get(configPath);
        if (!Files.exists(path)) {
            log.warn("Config file not found: {}", configPath);
            return YAML_MAPPER.createObjectNode();
        }

        try (InputStream is = Files.newInputStream(path)) {
            return YAML_MAPPER.readTree(is);
        } catch (IOException e) {
            log.error("Failed to load config from: {}", configPath, e);
            return YAML_MAPPER.createObjectNode();
        }
    }

    private static <T> T loadSection(JsonNode root, String sectionName, Class<T> configClass) {
        JsonNode section = root.get(sectionName);
        if (section == null || section.isMissingNode()) {
            log.warn("Section '{}' not found in config, using defaults", sectionName);
            return createDefault(configClass);
        }

        try {
            T config = YAML_MAPPER.treeToValue(section, configClass);
            log.info("Loaded {} config section", sectionName);
            return config;
        } catch (IOException e) {
            log.error("Failed to parse {} config section", sectionName, e);
            return createDefault(configClass);
        }
    }

    public static <T> T loadConfigFromResource(String resourcePath, Class<T> configClass) {
        try (InputStream is = ConfigLoader.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (is == null) {
                log.warn("Resource not found: {}, using defaults", resourcePath);
                return createDefault(configClass);
            }
            JsonNode root = YAML_MAPPER.readTree(is);
            String sectionName = getSectionName(configClass);
            return loadSection(root, sectionName, configClass);
        } catch (IOException e) {
            log.error("Failed to load config from resource: {}", resourcePath, e);
            return createDefault(configClass);
        }
    }

    private static String getSectionName(Class<?> configClass) {
        if (configClass == TrainingConfig.class) return "training";
        if (configClass == GAConfig.class) return "ga";
        if (configClass == CascadeConfig.class) return "cascade";
        if (configClass == AppConfig.class) return "app";
        return configClass.getSimpleName().toLowerCase();
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
            // Wrap config in a section for proper loading
            String sectionName = getSectionName(config.getClass());
            JsonNode root = YAML_MAPPER.valueToTree(config);
            JsonNode wrapper = YAML_MAPPER.createObjectNode().set(sectionName, root);
            YAML_MAPPER.writerWithDefaultPrettyPrinter()
                    .writeValue(Files.newOutputStream(path), wrapper);
            log.info("Saved config to: {}", outputPath);
        } catch (IOException e) {
            log.error("Failed to save config to: {}", outputPath, e);
            throw new IllegalStateException("Cannot save config", e);
        }
    }
}