function linreg(x,y; learning_rate=0.05, epochs=2000)
    #slope, intercept, and number of data points
    m=0.0
    b=0.0
    n=length(y)
    for epoch in 1:epochs
        m_gradient=0.0
        b_gradient=0.0

        #calculate gradients for all data point
        for i in 1:n
            y_pred=m*x[i]+b
            error=y_pred-y[i]
            m_gradient+=(2/n)*x[i]*error
            b_gradient+=(2/n)*error
        end
        m -= learning_rate * m_gradient
        b -= learning_rate * b_gradient
    end
    return m, b
end

function predict(x, m, b)
    return m*x+b
end

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [7.1, 8.9, 11.2, 13.0, 15.1]
m, b = linreg(x, y)
println("Line Equation: y = ", m, "x + ", b)

x_new = [6.0,7.5]
y_pred = predict.(x_new, m, b)
println("Predictions: ", y_pred)


#CPU EFFICIENCY
# mode = cpu, N = 1000
# compile time: 7.62 s
# result (m, b) = (1.9793748632642587, 5.058678125574738)
# run times (s): [0.002, 0.0013, 0.0013, 0.0013, 0.0013]
# best: 0.0013 s

# mode = cpu, N = 100000
# compile time: 7.85 s
# result (m, b) = (1.9976721331440086, 5.0066270491535105)
# run times (s): [0.2161, 0.218, 0.2193, 0.2132, 0.2154]
# best: 0.2132 s

#GPU EFFICIENCY
# mode = gpu, N = 1000
# compile time: 8.3 s
# result (m, b) = (1.9793748632642587, 5.058678125574738)
# run times (s): [0.0133, 0.0132, 0.0131, 0.0133, 0.0131]
# best: 0.0131 s

# mode = gpu, N = 100000
# compile time: 7.28 s
# result (m, b) = (1.9976721331440086, 5.0066270491535105)
# run times (s): [0.0215, 0.0214, 0.0215, 0.0215, 0.0215]
# best: 0.0214 s