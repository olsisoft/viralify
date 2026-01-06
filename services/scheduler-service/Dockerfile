FROM gradle:8.5-jdk21-alpine AS builder

WORKDIR /app
COPY build.gradle settings.gradle* ./
COPY src src
RUN gradle bootJar --no-daemon

FROM eclipse-temurin:21-jre-alpine

WORKDIR /app
COPY --from=builder /app/build/libs/*.jar app.jar

EXPOSE 8082

ENTRYPOINT ["java", "-jar", "app.jar"]
