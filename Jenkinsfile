pipeline {
  agent any
  environment {
    REGISTRY = "your-registry/aiml-inference"
  }
  stages {
    stage('Checkout') { steps { checkout scm } }
    stage('Unit Tests') {
      steps { sh 'python -m pytest -q service/tests' }
    }
    stage('Build') {
      steps { sh 'docker build -t ${REGISTRY}:${BUILD_NUMBER} service' }
    }
    stage('Push') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'docker-creds', usernameVariable:'USER', passwordVariable:'PASS')]) {
          sh 'echo $PASS | docker login -u $USER --password-stdin'
          sh 'docker push ${REGISTRY}:${BUILD_NUMBER}'
        }
      }
    }
    stage('Deploy') {
      steps {
        withCredentials([file(credentialsId:'kubeconfig', variable:'KUBECONFIG')]) {
          sh "kubectl set image deployment/inference-deployment inference=${REGISTRY}:${BUILD_NUMBER} -n default || kubectl apply -f deploy/k8s-deployment.yaml"
        }
      }
    }
  }
}
