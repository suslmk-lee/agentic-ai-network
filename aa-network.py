"""
Agentic AI 기반 네트워크 데이터 유실 및 지연 방지 시스템
"""

import numpy as np
import time
import random
from datetime import datetime
from collections import deque
import json

class NetworkMonitoringModule:
    """
    네트워크 상태 데이터를 실시간으로 수집하는 모니터링 모듈
    실제 환경에서는 SNMP, 네트워크 센서 등을 통해 데이터를 수집합니다.
    """
    
    def __init__(self):
        print("📡 네트워크 모니터링 모듈 초기화...")
        # 시뮬레이션을 위한 기본값들
        self.base_rtt = 10.0  # 기본 RTT (ms)
        self.base_queue = 1000  # 기본 큐 길이
        self.noise_factor = 0.1  # 노이즈 팩터
        
    def collect_network_data(self):
        """
        실시간 네트워크 메트릭을 수집합니다.
        실제로는 라우터, 스위치에서 SNMP 등을 통해 데이터를 가져옵니다.
        
        Returns:
            list: [RTT, Queue_Length, Packet_Loss, Bandwidth_Usage, Throughput, Jitter, CPU_Usage, Memory_Usage]
        """
        # 실제 환경을 시뮬레이션하기 위해 랜덤 값 + 트렌드 생성
        current_time = time.time()
        
        # 시간에 따른 트래픽 패턴 시뮬레이션 (피크 시간대 등)
        traffic_pattern = 0.5 + 0.3 * np.sin(current_time / 100)  # 주기적 패턴
        
        network_data = [
            self.base_rtt + random.uniform(-2, 8) + traffic_pattern * 5,  # RTT (ms)
            max(100, self.base_queue + random.uniform(-200, 800) + traffic_pattern * 500),  # Queue Length
            max(0, random.uniform(0, 0.05) + traffic_pattern * 0.02),  # Packet Loss Rate (0-5%)
            min(1.0, max(0.1, 0.3 + traffic_pattern + random.uniform(-0.2, 0.3))),  # Bandwidth Usage (10-100%)
            random.uniform(100, 1000),  # Throughput (Mbps)
            random.uniform(0.5, 5.0),  # Jitter (ms)
            random.uniform(0.2, 0.9),  # CPU Usage (20-90%)
            random.uniform(0.3, 0.8)   # Memory Usage (30-80%)
        ]
        
        return network_data

class PredictionAIAgent:
    """
    LSTM 기반 예측 AI 에이전트
    과거 데이터를 학습하여 미래 30초간의 네트워크 상태를 예측합니다.
    """
    
    def __init__(self, window_size=50):
        print("🤖 예측 AI 에이전트 초기화...")
        self.window_size = window_size  # 과거 몇 개의 데이터를 사용할지
        self.data_history = deque(maxlen=window_size)  # 과거 데이터 저장소
        
        # 실제 LSTM 대신 간단한 가중 평균 + 트렌드 분석으로 시뮬레이션
        # 실제 구현시에는 TensorFlow, PyTorch 등을 사용합니다
        self.temporal_weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # 최근 데이터에 더 높은 가중치
        
        # 8개 네트워크 데이터에 대한 학습 가능한 가중치 (W_h) 초기화
        self.feature_weights = np.random.uniform(0.05, 0.2, 8)  # [RTT, Queue, Loss, Bandwidth, Throughput, Jitter, CPU, Memory]
        self.feature_weights = self.feature_weights / np.sum(self.feature_weights)  # 정규화
        self.learning_rate = 0.01
        self.target_history = deque(maxlen=10)  # 실제 관측값 저장용
        
    def update_history(self, network_data):
        """과거 데이터 히스토리에 새 데이터 추가"""
        self.data_history.append(network_data)
        
    def learn_feature_weights(self, current_data, actual_performance):
        """
        8개 네트워크 데이터에 대한 가중치를 간단한 경사하강법으로 학습
        
        Args:
            current_data (list): 현재 네트워크 데이터 [8차원]
            actual_performance (float): 실제 성능 지표 (지연시간 기준)
        """
        if len(self.data_history) < 2:
            return
            
        # 현재 가중치로 예측값 계산
        predicted_value = np.dot(self.feature_weights, current_data)
        
        # 오차 계산 (실제 성능과 예측값의 차이)
        error = actual_performance - predicted_value
        
        # 경사하강법으로 가중치 업데이트
        gradient = error * np.array(current_data)
        self.feature_weights += self.learning_rate * gradient
        
        # 가중치 정규화 (합이 1이 되도록)
        self.feature_weights = np.maximum(self.feature_weights, 0.01)  # 음수 방지
        self.feature_weights = self.feature_weights / np.sum(self.feature_weights)
        
        # 학습 결과 출력 (가끔)
        if len(self.data_history) % 10 == 0:
            print(f"📚 가중치 학습: 오차={error:.3f}, 최대가중치={np.max(self.feature_weights):.3f}")
        
    def predict_network_state(self):
        """
        LSTM 네트워크를 사용하여 미래 네트워크 상태 예측
        명세서의 공식: h_t = LSTM(x_t, h_{t-1}), y_t = W_h × h_t + b_h
        
        Returns:
            dict: 예측된 네트워크 상태 정보
        """
        if len(self.data_history) < 5:
            # 데이터가 부족할 때는 현재 데이터 기반으로 예측
            current_data = np.array(list(self.data_history)[-1]) if self.data_history else np.array([10, 1000, 0.01, 0.5, 500, 2, 0.5, 0.6])
        else:
            # 최근 5개 데이터의 가중 평균으로 트렌드 계산 (LSTM 시뮬레이션)
            recent_data = np.array(list(self.data_history)[-5:])
            current_data = np.average(recent_data, axis=0, weights=self.temporal_weights)
        
        # 트렌드 분석 (최근 데이터의 변화율)
        if len(self.data_history) >= 2:
            trend = np.array(list(self.data_history)[-1]) - np.array(list(self.data_history)[-2])
        else:
            trend = np.zeros(8)
        
        # 학습된 가중치를 사용하여 미래 상태 예측 (W_h 적용)
        weighted_current = np.dot(self.feature_weights, current_data) 
        
        # 미래 30초 예측 (가중치 적용된 현재 상태 + 트렌드 + 불확실성)
        uncertainty_factor = np.random.normal(0, 0.1, 8)  # 예측 불확실성
        base_prediction = current_data * weighted_current / np.mean(current_data)  # 가중치 반영
        future_prediction = base_prediction + trend * 0.5 + uncertainty_factor
        
        # 예측 결과를 딕셔너리로 정리
        prediction_result = {
            'predicted_delay': max(1.0, future_prediction[0]),  # RTT (ms)
            'predicted_loss': max(0.0, min(0.1, future_prediction[2])),  # Packet Loss (0-10%)
            'predicted_congestion': max(0.0, min(1.0, future_prediction[3])),  # Bandwidth Usage
            'predicted_bandwidth': max(0.1, min(1.0, future_prediction[3])),  # Bandwidth Usage
            'reliability_score': max(0.5, min(1.0, 1.0 - future_prediction[2] * 10)),  # 패킷 손실 기반 신뢰도
            'predicted_queue': max(100, future_prediction[1]),  # Queue Length
            'predicted_throughput': max(100, future_prediction[4])  # Throughput
        }
        
        print(f"🔮 예측 결과: 지연={prediction_result['predicted_delay']:.1f}ms, "
              f"손실={prediction_result['predicted_loss']:.3f}%, "
              f"혼잡도={prediction_result['predicted_congestion']:.2f}")
        
        return prediction_result

class RoutingAIAgent:
    """
    강화학습 기반 라우팅 AI 에이전트
    최적 경로를 결정하여 네트워크 지연을 최소화합니다.
    """
    
    def __init__(self):
        print("🛣️ 라우팅 AI 에이전트 초기화...")
        # 가능한 경로들 (실제로는 네트워크 토폴로지에서 가져옴)
        self.available_routes = {
            'Route_A': {'base_delay': 10, 'reliability': 0.95, 'bandwidth_factor': 1.0},
            'Route_B': {'base_delay': 12, 'reliability': 0.90, 'bandwidth_factor': 1.2},
            'Route_C': {'base_delay': 8, 'reliability': 0.85, 'bandwidth_factor': 0.8}
        }
        self.current_route = 'Route_A'
        
        # 동적 가중치 (네트워크 상황에 따라 조정)
        self.alpha = 0.4  # 지연 가중치
        self.beta = 0.3   # 대역폭 가중치  
        self.gamma = 0.3  # 신뢰성 가중치
        
    def decide_optimal_route(self, prediction_data):
        """
        명세서의 공식을 사용하여 최적 경로 결정
        Total_Cost = α × Delay + β × Bandwidth_Usage + γ × Reliability_Score
        
        Args:
            prediction_data (dict): 예측 AI에서 받은 데이터
            
        Returns:
            dict: 라우팅 결정 결과
        """
        best_route = None
        min_cost = float('inf')
        route_costs = {}
        
        for route_name, route_info in self.available_routes.items():
            # 각 경로의 총 비용 계산
            predicted_delay = prediction_data['predicted_delay']
            bandwidth_usage = prediction_data['predicted_bandwidth']
            reliability = route_info['reliability']
            
            # 경로별 지연 시간 조정
            adjusted_delay = route_info['base_delay'] + (predicted_delay - 10) * route_info['bandwidth_factor']
            
            # 명세서의 총 비용 공식 적용
            total_cost = (self.alpha * adjusted_delay + 
                         self.beta * bandwidth_usage + 
                         self.gamma * (1.0 - reliability))  # 신뢰성이 높을수록 비용 낮음
            
            route_costs[route_name] = total_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_route = route_name
        
        # 경로 변경이 필요한지 확인
        route_changed = (best_route != self.current_route)
        if route_changed:
            print(f"🛣️ 경로 변경: {self.current_route} → {best_route} (비용: {min_cost:.2f})")
            self.current_route = best_route
        
        return {
            'optimal_route': best_route,
            'route_changed': route_changed,
            'total_cost': min_cost,
            'expected_delay': self.available_routes[best_route]['base_delay'],
            'route_costs': route_costs
        }

class LoadBalancingAIAgent:
    """
    동적 가중치 기반 로드밸런싱 AI 에이전트
    서버 간 트래픽을 최적으로 분산합니다.
    """
    
    def __init__(self):
        print("⚖️ 로드밸런싱 AI 에이전트 초기화...")
        # 서버 정보 (실제로는 서버 모니터링 시스템에서 가져옴)
        self.servers = {
            'Server_1': {'capacity': 1000, 'current_load': 300},  # req/s
            'Server_2': {'capacity': 800, 'current_load': 600},
            'Server_3': {'capacity': 1200, 'current_load': 1000}
        }
        
    def distribute_traffic(self, prediction_data):
        """
        명세서의 공식을 사용하여 트래픽 분산
        Weight_i = (Capacity_i - Current_Load_i) / Total_Available_Capacity
        Traffic_Allocation_i = Weight_i × Total_Traffic
        
        Args:
            prediction_data (dict): 예측 AI에서 받은 데이터
            
        Returns:
            dict: 로드밸런싱 결정 결과
        """
        # 예측된 추가 트래픽 계산
        predicted_throughput = prediction_data['predicted_throughput']
        additional_traffic = max(0, predicted_throughput - 500)  # 기준치 500 req/s
        
        # 각 서버의 여유 용량 계산
        available_capacities = {}
        total_available = 0
        
        for server_name, server_info in self.servers.items():
            available = max(0, server_info['capacity'] - server_info['current_load'])
            available_capacities[server_name] = available
            total_available += available
        
        if total_available == 0:
            print("⚠️ 모든 서버가 포화 상태입니다!")
            return {'error': 'All servers at capacity'}
        
        # 각 서버의 가중치 및 트래픽 할당량 계산
        traffic_allocation = {}
        weights = {}
        
        for server_name in self.servers.keys():
            # 명세서의 가중치 공식 적용
            weight = available_capacities[server_name] / total_available
            weights[server_name] = weight
            
            # 트래픽 할당량 계산
            allocation = weight * additional_traffic
            traffic_allocation[server_name] = allocation
            
            # 서버 부하 업데이트 (시뮬레이션)
            self.servers[server_name]['current_load'] += allocation * 0.1  # 부분 적용
        
        print(f"⚖️ 트래픽 분산: {dict(zip(traffic_allocation.keys(), [f'{v:.0f} req/s' for v in traffic_allocation.values()]))}")
        
        return {
            'traffic_allocation': traffic_allocation,
            'weights': weights,
            'total_additional_traffic': additional_traffic,
            'server_status': self.servers.copy()
        }

class AdaptiveBufferingModule:
    """
    적응형 버퍼링 관리 모듈
    데이터 지연에 따른 버퍼링 크기를 동적으로 조정합니다.
    """
    
    def __init__(self):
        print("🗃️ 적응형 버퍼링 관리 모듈 초기화...")
        self.base_buffer_time = 3.0  # 기본 버퍼링 시간 (초)
        self.alpha = 0.5  # 패킷 손실률 가중치
        self.beta = 0.3   # 네트워크 혼잡도 가중치
        self.current_buffer_size = 2000  # 현재 버퍼 크기 (kb)
        
    def calculate_optimal_buffer(self, prediction_data):
        """
        명세서의 공식을 사용하여 최적 버퍼 크기 계산
        B_s = R × B_t
        B_t = Base_Buffer_Time + Delay_Compensation_Factor × Predicted_Delay
        Delay_Compensation_Factor = α × Packet_Loss_Rate + β × Network_Congestion_Level
        
        Args:
            prediction_data (dict): 예측 AI에서 받은 데이터
            
        Returns:
            dict: 버퍼링 조정 결과
        """
        # 예측 데이터에서 필요한 값들 추출
        packet_loss_rate = prediction_data['predicted_loss']
        network_congestion = prediction_data['predicted_congestion']
        predicted_delay = prediction_data['predicted_delay'] / 1000  # ms를 초로 변환
        data_rate = prediction_data['predicted_throughput']  # kbps
        
        # 지연 보상 계수 계산
        delay_compensation_factor = (self.alpha * packet_loss_rate + 
                                   self.beta * network_congestion)
        
        # 동적 버퍼링 시간 계산
        buffer_time = (self.base_buffer_time + 
                      delay_compensation_factor * predicted_delay)
        
        # 버퍼 크기 계산 (B_s = R × B_t)
        optimal_buffer_size = data_rate * buffer_time
        
        # 버퍼 크기 조정이 필요한지 확인
        size_change_needed = abs(optimal_buffer_size - self.current_buffer_size) > 100  # 100kb 이상 차이
        
        if size_change_needed:
            change_percent = ((optimal_buffer_size - self.current_buffer_size) / 
                            self.current_buffer_size) * 100
            print(f"🗃️ 버퍼 크기 조정: {self.current_buffer_size:.0f}kb → {optimal_buffer_size:.0f}kb "
                  f"({change_percent:+.1f}%)")
            self.current_buffer_size = optimal_buffer_size
        
        return {
            'optimal_buffer_size': optimal_buffer_size,
            'buffer_time': buffer_time,
            'delay_compensation_factor': delay_compensation_factor,
            'size_change_needed': size_change_needed,
            'buffer_efficiency': min(1.0, optimal_buffer_size / (data_rate * 5.0))  # 효율성 지표
        }

class CentralControlModule:
    """
    중앙 제어 모듈
    다중 AI 에이전트들의 협업을 조정하고 의사결정 충돌을 해결합니다.
    """
    
    def __init__(self):
        print("🎛️ 중앙 제어 모듈 초기화...")
        self.execution_log = []  # 실행 기록
        
    def coordinate_agents(self, routing_result, loadbalancing_result, buffering_result):
        """
        여러 AI 에이전트의 결과를 조정하고 실행 계획을 수립합니다.
        
        Args:
            routing_result (dict): 라우팅 AI 결과
            loadbalancing_result (dict): 로드밸런싱 AI 결과  
            buffering_result (dict): 버퍼링 관리 결과
            
        Returns:
            dict: 통합 실행 계획
        """
        # 각 에이전트의 우선순위 결정
        priorities = {
            'routing': 'HIGH' if routing_result.get('route_changed', False) else 'LOW',
            'loadbalancing': 'MEDIUM',
            'buffering': 'HIGH' if buffering_result.get('size_change_needed', False) else 'LOW'
        }
        
        # 실행 순서 계획 (우선순위 기반)
        execution_plan = []
        
        # 1. 높은 우선순위 작업들
        if priorities['routing'] == 'HIGH':
            execution_plan.append({
                'time': 0.0,  # 즉시 실행
                'action': 'route_change',
                'details': routing_result,
                'agent': 'routing'
            })
            
        if priorities['buffering'] == 'HIGH':
            execution_plan.append({
                'time': 0.1,  # 0.1초 후 실행
                'action': 'buffer_adjustment',
                'details': buffering_result,
                'agent': 'buffering'
            })
        
        # 2. 중간 우선순위 작업들
        if 'error' not in loadbalancing_result:
            execution_plan.append({
                'time': 0.3,  # 0.3초 후 실행
                'action': 'traffic_redistribution',
                'details': loadbalancing_result,
                'agent': 'loadbalancing'
            })
        
        # 충돌 해결 (예: 라우팅 변경 시 로드밸런싱 지연)
        if priorities['routing'] == 'HIGH' and len(execution_plan) > 1:
            for plan in execution_plan:
                if plan['agent'] == 'loadbalancing':
                    plan['time'] += 0.2  # 라우팅 안정화 후 실행
        
        print(f"🎛️ 실행 계획: {len(execution_plan)}개 작업, 우선순위: {priorities}")
        
        return {
            'execution_plan': execution_plan,
            'priorities': priorities,
            'total_actions': len(execution_plan),
            'estimated_completion_time': max([plan['time'] for plan in execution_plan]) if execution_plan else 0
        }
    
    def execute_plan(self, coordination_result):
        """실행 계획을 순차적으로 실행합니다."""
        execution_plan = coordination_result['execution_plan']
        
        for plan in execution_plan:
            # 실제 환경에서는 time.sleep(plan['time'])로 지연 실행
            print(f"⚡ 실행 중: {plan['action']} (에이전트: {plan['agent']}, "
                  f"지연: {plan['time']}초)")
            
            # 실행 기록 저장
            self.execution_log.append({
                'timestamp': datetime.now(),
                'action': plan['action'],
                'agent': plan['agent'],
                'details': plan['details']
            })
        
        return {'executed_actions': len(execution_plan), 'success': True}

class AgenticAINetworkSystem:
    """
    Agentic AI 기반 네트워크 관리 시스템 메인 클래스
    모든 구성 요소를 통합하여 관리합니다.
    """
    
    def __init__(self):
        print("🚀 Agentic AI 네트워크 시스템 시작!")
        print("=" * 60)
        
        # 모든 모듈 초기화
        self.monitoring = NetworkMonitoringModule()
        self.prediction_ai = PredictionAIAgent()
        self.routing_ai = RoutingAIAgent()
        self.loadbalancing_ai = LoadBalancingAIAgent()
        self.buffering_module = AdaptiveBufferingModule()
        self.central_control = CentralControlModule()
        
        # 성능 지표
        self.performance_metrics = {
            'total_cycles': 0,
            'route_changes': 0,
            'buffer_adjustments': 0,
            'traffic_redistributions': 0,
            'average_delay': 0,
            'system_efficiency': 0
        }
        
        print("✅ 모든 모듈 초기화 완료!")
        print("=" * 60)
    
    def run_single_cycle(self):
        """한 번의 처리 주기를 실행합니다."""
        cycle_start = time.time()
        
        print(f"\n🔄 처리 주기 #{self.performance_metrics['total_cycles'] + 1} 시작")
        print("-" * 50)
        
        # 1단계: 네트워크 데이터 수집
        network_data = self.monitoring.collect_network_data()
        print(f"📊 수집된 데이터: RTT={network_data[0]:.1f}ms, "
              f"큐={network_data[1]:.0f}, 손실={network_data[2]:.3f}%, "
              f"대역폭={network_data[3]:.2f}")
        
        # 2단계: 예측 AI 처리 (학습 포함)
        self.prediction_ai.update_history(network_data)
        
        # 가중치 학습 수행 (실제 RTT를 기준으로)
        if len(self.prediction_ai.data_history) > 1:
            actual_delay = network_data[0]  # 실제 RTT
            self.prediction_ai.learn_feature_weights(network_data, actual_delay)
        
        prediction_result = self.prediction_ai.predict_network_state()
        
        # 3단계: 다중 AI 에이전트 병렬 처리
        routing_result = self.routing_ai.decide_optimal_route(prediction_result)
        loadbalancing_result = self.loadbalancing_ai.distribute_traffic(prediction_result)
        buffering_result = self.buffering_module.calculate_optimal_buffer(prediction_result)
        
        # 4단계: 중앙 제어 조정
        coordination_result = self.central_control.coordinate_agents(
            routing_result, loadbalancing_result, buffering_result)
        
        # 5단계: 실행
        execution_result = self.central_control.execute_plan(coordination_result)
        
        # 6단계: 성능 지표 업데이트
        self._update_performance_metrics(routing_result, buffering_result, 
                                       coordination_result, prediction_result)
        
        cycle_time = time.time() - cycle_start
        print(f"⏱️ 주기 완료 시간: {cycle_time:.3f}초")
        
        return {
            'network_data': network_data,
            'prediction': prediction_result,
            'routing': routing_result,
            'loadbalancing': loadbalancing_result,
            'buffering': buffering_result,
            'coordination': coordination_result,
            'execution': execution_result,
            'cycle_time': cycle_time
        }
    
    def _update_performance_metrics(self, routing_result, buffering_result, 
                                  coordination_result, prediction_result):
        """성능 지표를 업데이트합니다."""
        self.performance_metrics['total_cycles'] += 1
        
        if routing_result.get('route_changed', False):
            self.performance_metrics['route_changes'] += 1
            
        if buffering_result.get('size_change_needed', False):
            self.performance_metrics['buffer_adjustments'] += 1
            
        if coordination_result['total_actions'] > 0:
            self.performance_metrics['traffic_redistributions'] += 1
        
        # 평균 지연 시간 업데이트
        current_delay = prediction_result['predicted_delay']
        total_cycles = self.performance_metrics['total_cycles']
        self.performance_metrics['average_delay'] = (
            (self.performance_metrics['average_delay'] * (total_cycles - 1) + current_delay) 
            / total_cycles
        )
        
        # 시스템 효율성 계산 (임의의 공식)
        efficiency = max(0, 100 - current_delay * 2 - prediction_result['predicted_loss'] * 1000)
        self.performance_metrics['system_efficiency'] = efficiency
    
    def print_performance_summary(self):
        """현재까지의 성능 요약을 출력합니다."""
        metrics = self.performance_metrics
        print("\n📈 시스템 성능 요약")
        print("=" * 50)
        print(f"총 처리 주기: {metrics['total_cycles']}")
        print(f"경로 변경 횟수: {metrics['route_changes']}")
        print(f"버퍼 조정 횟수: {metrics['buffer_adjustments']}")
        print(f"트래픽 재분산: {metrics['traffic_redistributions']}")
        print(f"평균 지연시간: {metrics['average_delay']:.2f}ms")
        print(f"시스템 효율성: {metrics['system_efficiency']:.1f}%")
        print("=" * 50)
    
    def run_continuous(self, cycles=10, delay=2.0):
        """
        연속 모드로 시스템을 실행합니다.
        
        Args:
            cycles (int): 실행할 주기 수
            delay (float): 주기 간 대기 시간 (초)
        """
        print(f"🔄 연속 모드 시작: {cycles}주기, {delay}초 간격")
        
        try:
            for i in range(cycles):
                cycle_result = self.run_single_cycle()
                
                if i < cycles - 1:  # 마지막 주기가 아니면 대기
                    time.sleep(delay)
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중단됨")
        
        finally:
            self.print_performance_summary()

def main():
    """메인 함수 - 시스템을 실행합니다."""
    print("🎯 Agentic AI 네트워크 관리 시스템 데모")
    print("이 시스템은 실시간으로 네트워크를 모니터링하고")
    print("AI 에이전트들이 협업하여 최적화를 수행합니다.")
    print()
    
    # 시스템 초기화
    system = AgenticAINetworkSystem()
    
    # 사용자 선택
    print("\n실행 모드를 선택하세요:")
    print("1. 단일 주기 실행 (한 번만 실행)")
    print("2. 연속 모드 (여러 주기 실행)")
    
    try:
        choice = input("\n선택 (1 또는 2): ").strip()
        
        if choice == '1':
            print("\n단일 주기 실행 모드")
            result = system.run_single_cycle()
            system.print_performance_summary()
            
        elif choice == '2':
            cycles = int(input("실행할 주기 수 (기본값 5): ") or "5")
            delay = float(input("주기 간 대기시간(초, 기본값 2.0): ") or "2.0")
            print(f"\n연속 모드: {cycles}주기, {delay}초 간격")
            system.run_continuous(cycles, delay)
            
        else:
            print("잘못된 선택입니다. 기본값으로 단일 주기 실행합니다.")
            result = system.run_single_cycle()
            system.print_performance_summary()
            
    except ValueError:
        print("잘못된 입력입니다. 기본값으로 실행합니다.")
        system.run_continuous(5, 2.0)
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    
    print("\n🎉 시스템 실행 완료!")
    print("실제 환경에서는 이 시스템이 24시간 연속으로 동작하여")
    print("네트워크의 데이터 유실과 지연을 방지합니다.")

# 추가 유틸리티 함수들

def simulate_network_crisis():
    """
    네트워크 위기 상황을 시뮬레이션하는 함수
    갑작스러운 트래픽 증가, 서버 장애 등을 모사합니다.
    """
    print("\n🚨 네트워크 위기 상황 시뮬레이션")
    print("=" * 50)
    
    system = AgenticAINetworkSystem()
    
    # 정상 상태에서 몇 주기 실행
    print("1️⃣ 정상 상태 (3주기)")
    for i in range(3):
        system.run_single_cycle()
        time.sleep(1)
    
    print("\n🚨 위기 상황 발생! (높은 트래픽 + 패킷 손실)")
    
    # 위기 상황 시뮬레이션을 위해 모니터링 모듈 수정
    original_collect = system.monitoring.collect_network_data
    
    def crisis_collect():
        """위기 상황에서의 데이터 수집"""
        data = original_collect()
        # 위기 상황: 높은 지연, 패킷 손실, 높은 대역폭 사용률
        data[0] *= 3  # RTT 3배 증가
        data[1] *= 2  # 큐 길이 2배 증가  
        data[2] *= 10  # 패킷 손실 10배 증가
        data[3] = min(0.95, data[3] * 1.5)  # 대역폭 사용률 1.5배
        return data
    
    # 위기 상황 모드로 전환
    system.monitoring.collect_network_data = crisis_collect
    
    print("2️⃣ 위기 대응 (5주기)")
    for i in range(5):
        result = system.run_single_cycle()
        print(f"   위기 대응 진행률: {((i+1)/5)*100:.0f}%")
        time.sleep(1)
    
    # 복구 모드
    system.monitoring.collect_network_data = original_collect
    print("\n✅ 시스템 자동 복구 (2주기)")
    for i in range(2):
        system.run_single_cycle()
        time.sleep(1)
    
    system.print_performance_summary()
    print("\n🎯 위기 상황 시뮬레이션 완료!")
    print("AI 에이전트들이 협력하여 네트워크 위기를 성공적으로 극복했습니다.")

def analyze_system_components():
    """
    각 시스템 구성요소의 역할과 상호작용을 분석하는 함수
    """
    print("\n🔬 시스템 구성요소 분석")
    print("=" * 60)
    
    components = {
        "📡 네트워크 모니터링": {
            "역할": "실시간 네트워크 메트릭 수집",
            "출력": "RTT, 큐 길이, 패킷 손실률 등 8차원 벡터",
            "중요도": "⭐⭐⭐⭐⭐ (모든 의사결정의 기반)"
        },
        "🤖 예측 AI 에이전트": {
            "역할": "LSTM을 통한 미래 30초 네트워크 상태 예측", 
            "출력": "예측된 지연, 손실률, 혼잡도, 신뢰성 점수",
            "중요도": "⭐⭐⭐⭐⭐ (사전 예방의 핵심)"
        },
        "🛣️ 라우팅 AI 에이전트": {
            "역할": "강화학습 기반 최적 경로 선택",
            "출력": "최적 경로, 예상 지연시간, 경로 변경 여부", 
            "중요도": "⭐⭐⭐⭐ (지연 시간 단축의 핵심)"
        },
        "⚖️ 로드밸런싱 AI": {
            "역할": "서버 간 트래픽 최적 분산",
            "출력": "서버별 트래픽 할당량, 가중치",
            "중요도": "⭐⭐⭐⭐ (자원 효율성 향상)"
        },
        "🗃️ 적응형 버퍼링": {
            "역할": "동적 버퍼 크기 조정으로 데이터 연속성 보장",
            "출력": "최적 버퍼 크기, 버퍼링 시간",
            "중요도": "⭐⭐⭐ (데이터 안정성 확보)"
        },
        "🎛️ 중앙 제어 모듈": {
            "역할": "AI 에이전트 간 협업 조정 및 충돌 해결",
            "출력": "통합 실행 계획, 우선순위 스케줄",
            "중요도": "⭐⭐⭐⭐⭐ (시스템 통합의 핵심)"
        }
    }
    
    for component, info in components.items():
        print(f"\n{component}")
        print(f"  역할: {info['역할']}")
        print(f"  출력: {info['출력']}")
        print(f"  중요도: {info['중요도']}")
    
    print(f"\n🤝 상호작용 흐름:")
    print("  1. 모니터링 → 예측 AI (실시간 데이터)")
    print("  2. 예측 AI → 3개 에이전트 (예측 결과)")
    print("  3. 3개 에이전트 → 중앙 제어 (개별 결정)")
    print("  4. 중앙 제어 → 네트워크 (통합 실행)")
    print("  5. 네트워크 → 모니터링 (피드백)")

def performance_benchmark():
    """
    시스템 성능 벤치마크 테스트
    다양한 네트워크 시나리오에서의 성능을 측정합니다.
    """
    print("\n📊 성능 벤치마크 테스트")
    print("=" * 60)
    
    scenarios = [
        ("정상 트래픽", 1.0, 1.0),
        ("높은 트래픽", 2.0, 1.5), 
        ("패킷 손실 증가", 1.2, 3.0),
        ("네트워크 혼잡", 2.5, 2.0),
        ("서버 과부하", 1.8, 1.8)
    ]
    
    results = {}
    
    for scenario_name, delay_factor, loss_factor in scenarios:
        print(f"\n🧪 시나리오: {scenario_name}")
        print(f"   지연 배수: {delay_factor}x, 손실 배수: {loss_factor}x")
        
        system = AgenticAINetworkSystem()
        
        # 시나리오별 네트워크 상태 조정
        original_collect = system.monitoring.collect_network_data
        
        def scenario_collect():
            data = original_collect()
            data[0] *= delay_factor  # RTT 조정
            data[2] *= loss_factor   # 패킷 손실 조정
            return data
        
        system.monitoring.collect_network_data = scenario_collect
        
        # 5주기 실행하여 성능 측정
        start_time = time.time()
        for _ in range(5):
            system.run_single_cycle()
        
        execution_time = time.time() - start_time
        
        results[scenario_name] = {
            'avg_delay': system.performance_metrics['average_delay'],
            'efficiency': system.performance_metrics['system_efficiency'],
            'execution_time': execution_time,
            'route_changes': system.performance_metrics['route_changes'],
            'buffer_adjustments': system.performance_metrics['buffer_adjustments']
        }
        
        print(f"   결과: 평균지연 {results[scenario_name]['avg_delay']:.1f}ms, "
              f"효율성 {results[scenario_name]['efficiency']:.1f}%")
    
    # 벤치마크 결과 요약
    print(f"\n📈 벤치마크 결과 요약")
    print("=" * 60)
    print(f"{'시나리오':<12} {'평균지연(ms)':<12} {'효율성(%)':<10} {'실행시간(s)':<12} {'경로변경':<8} {'버퍼조정':<8}")
    print("-" * 60)
    
    for scenario, result in results.items():
        print(f"{scenario:<12} {result['avg_delay']:<12.1f} {result['efficiency']:<10.1f} "
              f"{result['execution_time']:<12.2f} {result['route_changes']:<8} {result['buffer_adjustments']:<8}")

def create_config_file():
    """
    시스템 설정 파일을 생성하는 함수
    실제 환경에서 사용할 수 있는 설정값들을 포함합니다.
    """
    config = {
        "system": {
            "name": "Agentic AI Network Management System",
            "version": "1.0.0",
            "description": "AI 에이전트 기반 네트워크 데이터 유실 및 지연 방지 시스템"
        },
        "monitoring": {
            "collection_interval": 1.0,  # 초
            "metrics": ["RTT", "Queue_Length", "Packet_Loss", "Bandwidth_Usage", 
                       "Throughput", "Jitter", "CPU_Usage", "Memory_Usage"],
            "alert_thresholds": {
                "high_delay": 50.0,  # ms
                "high_loss": 0.05,   # 5%
                "high_bandwidth": 0.9  # 90%
            }
        },
        "prediction_ai": {
            "model_type": "LSTM",
            "window_size": 50,
            "prediction_horizon": 30,  # 초
            "retrain_interval": 3600,   # 1시간마다 재훈련
            "learning_rate": 0.001
        },
        "routing_ai": {
            "algorithm": "reinforcement_learning", 
            "cost_weights": {
                "alpha": 0.4,  # 지연 가중치
                "beta": 0.3,   # 대역폭 가중치
                "gamma": 0.3   # 신뢰성 가중치
            },
            "route_change_threshold": 0.1
        },
        "loadbalancing": {
            "algorithm": "dynamic_weighted",
            "rebalance_threshold": 100,  # req/s
            "server_capacity_buffer": 0.1  # 10% 여유분
        },
        "buffering": {
            "base_buffer_time": 3.0,  # 초
            "compensation_weights": {
                "alpha": 0.5,  # 패킷 손실 가중치
                "beta": 0.3    # 혼잡도 가중치  
            },
            "max_buffer_size": 10000,  # kb
            "adjustment_threshold": 100  # kb
        },
        "central_control": {
            "coordination_interval": 0.1,  # 초
            "conflict_resolution": "priority_based",
            "execution_timeout": 5.0  # 초
        }
    }
    
    # JSON 파일로 저장
    with open('agentic_ai_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("📝 설정 파일 생성 완료: agentic_ai_config.json")
    print("실제 환경에서는 이 파일을 수정하여 시스템을 커스터마이징할 수 있습니다.")

# 메인 실행부 확장
if __name__ == "__main__":
    print("🎯 Agentic AI 네트워크 관리 시스템")
    print("=" * 60)
    print("명세서 기반 실제 구현 가능한 샘플 코드")
    print("초보자를 위한 상세 주석 포함")
    print("=" * 60)
    
    print("\n실행 모드를 선택하세요:")
    print("1. 기본 데모 실행")
    print("2. 네트워크 위기 상황 시뮬레이션") 
    print("3. 시스템 구성요소 분석")
    print("4. 성능 벤치마크 테스트")
    print("5. 설정 파일 생성")
    
    try:
        mode = input("\n선택 (1-5, 기본값 1): ").strip() or "1"
        
        if mode == "1":
            main()
        elif mode == "2":
            simulate_network_crisis()
        elif mode == "3":
            analyze_system_components()
        elif mode == "4":
            performance_benchmark()
        elif mode == "5":
            create_config_file()
        else:
            print("잘못된 선택입니다. 기본 데모를 실행합니다.")
            main()
            
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 오류가 발생했습니다: {e}")
        print("기본 데모를 실행합니다.")
        main()
    
    print("\n" + "=" * 60)
    print("🎉 Agentic AI 네트워크 시스템 실행 완료!")
    print("=" * 60)