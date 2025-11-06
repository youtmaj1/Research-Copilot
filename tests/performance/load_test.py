#!/usr/bin/env python3
"""
Load Testing Script for Research Copilot
========================================
Tests concurrent request handling and system stability
"""
import asyncio
import aiohttp
import time
import json
from concurrent.futures import ThreadPoolExecutor
import statistics

async def make_request(session, url, data=None):
    """Make async HTTP request"""
    start_time = time.time()
    try:
        if data:
            async with session.post(url, json=data) as response:
                result = await response.json()
                return {
                    'status': response.status,
                    'response_time': time.time() - start_time,
                    'success': response.status == 200,
                    'cached': result.get('cached', False)
                }
        else:
            async with session.get(url) as response:
                result = await response.json()
                return {
                    'status': response.status,
                    'response_time': time.time() - start_time,
                    'success': response.status == 200
                }
    except Exception as e:
        return {
            'status': 500,
            'response_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }

async def load_test():
    """Run load testing scenarios"""
    base_url = "http://localhost:8000"
    
    print("🚀 LOAD TESTING: Research Copilot System")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Health endpoint concurrent requests
        print("\n1. Health Endpoint Load Test (50 concurrent requests)")
        tasks = []
        for i in range(50):
            task = make_request(session, f"{base_url}/health")
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        success_count = sum(1 for r in results if r['success'])
        response_times = [r['response_time'] for r in results if r['success']]
        
        print(f"  ✅ Success Rate: {success_count}/50 ({success_count/50*100:.1f}%)")
        print(f"  ⚡ Total Time: {total_time:.2f}s")
        print(f"  📊 Avg Response: {statistics.mean(response_times):.3f}s")
        print(f"  📈 Throughput: {50/total_time:.1f} req/s")
        
        # Test 2: Stats endpoint load test
        print("\n2. Stats Endpoint Load Test (30 concurrent requests)")
        tasks = []
        for i in range(30):
            task = make_request(session, f"{base_url}/api/v1/stats")
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        success_count = sum(1 for r in results if r['success'])
        response_times = [r['response_time'] for r in results if r['success']]
        
        print(f"  ✅ Success Rate: {success_count}/30 ({success_count/30*100:.1f}%)")
        print(f"  ⚡ Total Time: {total_time:.2f}s")
        print(f"  📊 Avg Response: {statistics.mean(response_times):.3f}s")
        print(f"  📈 Throughput: {30/total_time:.1f} req/s")
        
        # Test 3: Query endpoint with caching (10 same queries)
        print("\n3. Query Caching Load Test (10 identical queries)")
        query_data = {"query": "What is machine learning?"}
        tasks = []
        for i in range(10):
            task = make_request(session, f"{base_url}/api/v1/query", query_data)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        success_count = sum(1 for r in results if r['success'])
        cached_count = sum(1 for r in results if r.get('cached', False))
        response_times = [r['response_time'] for r in results if r['success']]
        
        print(f"  ✅ Success Rate: {success_count}/10 ({success_count/10*100:.1f}%)")
        print(f"  💾 Cache Hits: {cached_count}/10 ({cached_count/10*100:.1f}%)")
        print(f"  ⚡ Total Time: {total_time:.2f}s")
        if response_times:
            print(f"  📊 Avg Response: {statistics.mean(response_times):.3f}s")
        print(f"  📈 Throughput: {10/total_time:.1f} req/s")

if __name__ == "__main__":
    asyncio.run(load_test())
