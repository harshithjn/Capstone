# Cross-Environment Runtime Prediction (Capstone Project)

This repository contains our **final-year capstone project**, focused on predicting **production runtime performance from development-time metrics**.

## What we are building

In real-world software development, code is tested on low-resource environments (laptops or small VMs) but deployed on powerful production servers. This mismatch makes it hard to predict runtime behavior and often leads to **SLA violations, deployment delays, and costly debugging**.

Our project builds a **machine learning–based framework** that:

- Collects runtime and system-level metrics (CPU, memory, I/O) during development runs
- Learns how these metrics translate to production environments
- Predicts production execution time **before deployment**
- Flags potential **SLA violations early** in the development cycle

## How it works (high level)

- We run **real, industry-standard workloads** (MLPerf, TPC-H, web benchmarks)
- Metrics are collected automatically using system profilers
- A cross-environment dataset (dev → prod) is created
- ML models are trained to forecast production runtime and SLA risk

## Why this matters

This project aims to make deployments **more reliable, SLA-aware, and cost-efficient**, and can be integrated into CI/CD pipelines for early performance validation.

> This work is based on our capstone research and dataset design outlined in the project documentation. :contentReference[oaicite:0]{index=0}
