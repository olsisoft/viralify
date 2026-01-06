package com.tiktok.platform.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.reactive.CorsWebFilter;
import org.springframework.web.cors.reactive.UrlBasedCorsConfigurationSource;
import java.util.Arrays;

@SpringBootApplication
public class ApiGatewayApplication {

    @Value("${auth-service.url:http://auth-service:8081}")
    private String authServiceUrl;

    @Value("${trend-service.url:http://trend-analyzer:8000}")
    private String trendServiceUrl;

    @Value("${content-service.url:http://content-generator:8001}")
    private String contentServiceUrl;

    @Value("${scheduler-service.url:http://scheduler-service:8082}")
    private String schedulerServiceUrl;

    @Value("${analytics-service.url:http://analytics-service:8002}")
    private String analyticsServiceUrl;

    public static void main(String[] args) {
        SpringApplication.run(ApiGatewayApplication.class, args);
    }

    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
                // Auth Service Routes
                .route("auth-service", r -> r
                        .path("/api/v1/auth/**")
                        .filters(f -> f.retry(3))
                        .uri(authServiceUrl))

                // Trend Analyzer Routes
                .route("trend-analyzer", r -> r
                        .path("/api/v1/trends/**")
                        .filters(f -> f.retry(2))
                        .uri(trendServiceUrl))

                // Content Generator Routes - with RewritePath to map /content/* to /*
                .route("content-generator", r -> r
                        .path("/api/v1/content/**")
                        .filters(f -> f
                                .rewritePath("/api/v1/content/(?<segment>.*)", "/api/v1/${segment}")
                                .retry(2))
                        .uri(contentServiceUrl))

                // Content Generator Direct Routes (no rewrite needed)
                .route("content-generator-direct", r -> r
                        .path("/api/v1/generate/**", "/api/v1/chat/**", "/api/v1/strategy/**")
                        .filters(f -> f.retry(2))
                        .uri(contentServiceUrl))

                // Scheduler Service Routes
                .route("scheduler-service", r -> r
                        .path("/api/v1/scheduler/**")
                        .filters(f -> f.retry(2))
                        .uri(schedulerServiceUrl))

                // Analytics Service Routes
                .route("analytics-service", r -> r
                        .path("/api/v1/analytics/**")
                        .filters(f -> f.retry(2))
                        .uri(analyticsServiceUrl))

                // Notification Service Routes
                .route("notification-service", r -> r
                        .path("/api/v1/notifications/**")
                        .uri("http://notification-service:8003"))

                // Health check endpoint
                .route("health", r -> r
                        .path("/health")
                        .filters(f -> f.setPath("/actuator/health"))
                        .uri("http://localhost:8080"))

                .build();
    }

    @Bean
    public CorsWebFilter corsWebFilter() {
        CorsConfiguration corsConfig = new CorsConfiguration();
        corsConfig.setAllowedOrigins(Arrays.asList("http://localhost:3000", "https://viraltok.app"));
        corsConfig.setMaxAge(3600L);
        corsConfig.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"));
        corsConfig.setAllowedHeaders(Arrays.asList("*"));
        corsConfig.setAllowCredentials(true);

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", corsConfig);

        return new CorsWebFilter(source);
    }
}
