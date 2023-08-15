#%load py_src/visualize_digit.py
# visualize single data instances
def visualize_digit(img_no=0):
    # img_no = 0 #change the number to display other examples

    first_number = x_train[img_no]
    plt.imshow(first_number, cmap='gray') # visualize the numbers in gray mode
    plt.show()
    print(f"correct number: {y_train[img_no]}")
