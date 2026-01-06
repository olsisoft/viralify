package com.tiktok.platform.auth.repository;

import com.tiktok.platform.auth.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface UserRepository extends JpaRepository<User, UUID> {
    Optional<User> findByEmail(String email);
    Optional<User> findByTiktokUserId(String tiktokUserId);
    boolean existsByEmail(String email);
    boolean existsByTiktokUserId(String tiktokUserId);
}
