
import json


# Load business configuration
with open("business_config.json", "r") as file:
    config = json.load(file)


def get_whatsapp_message(customer_name, order_id):
    return f"""
Hi {customer_name}! 👋

Thank you for your order #{order_id}.

Your order is currently placed with Cash on Delivery (COD).

Please confirm whether you will be available to receive it.

Reply:
YES - Confirm my order
NO - Cancel my order

Please respond within 24 hours.
Thank you!
"""


def get_prepaid_message(customer_name, order_id, order_value, discount):
    final_amount = order_value - discount

    return f"""
Hi {customer_name}! 👋

We have a special offer for your order #{order_id}!

Pay online in advance and get ₹{discount} OFF.

Original amount: ₹{order_value}
Discount: ₹{discount}
Prepaid amount: ₹{final_amount}

Reply:
YES - Accept prepaid offer
NO - Continue with COD

Thank you!
"""


def process_customer_reply(reply):
    reply = reply.strip().upper()

    if reply == "YES":
        return {
            "status": "CONFIRMED",
            "action": "Proceed with COD shipment",
            "message": "Customer confirmed the order."
        }

    elif reply == "NO":
        return {
            "status": "CANCELLED",
            "action": "Stop dispatch",
            "message": "Customer declined the order."
        }

    elif reply == "":
        return {
            "status": "NO RESPONSE",
            "action": "Hold for manual review",
            "message": "No response received within 24 hours."
        }

    else:
        return {
            "status": "UNCLEAR",
            "action": "Request a clear YES or NO reply",
            "message": "Customer response could not be understood."
        }


def process_prepaid_reply(reply, order_value, discount):
    reply = reply.strip().upper()

    if reply == "YES":
        final_amount = order_value - discount

        return {
            "status": "PREPAID ACCEPTED",
            "action": "Generate prepaid payment link",
            "message": "Customer accepted the prepaid offer.",
            "payment_amount": final_amount
        }

    elif reply == "NO":
        return {
            "status": "OFFER REJECTED",
            "action": "Continue with COD confirmation",
            "message": "Customer rejected the prepaid offer."
        }

    elif reply == "":
        return {
            "status": "NO RESPONSE",
            "action": "Hold for manual review",
            "message": "No response received within 24 hours."
        }

    else:
        return {
            "status": "UNCLEAR",
            "action": "Request a clear YES or NO reply",
            "message": "Customer response could not be understood."
        }


if __name__ == "__main__":

    customer_name = "Rahul"
    order_id = "ORD1001"
    order_value = 649

    # Read discount from business_config.json
    discount = config["prepaid_discount"]

    print("\n========== WHATSAPP SIMULATOR ==========")
    print("1. COD Confirmation")
    print("2. Prepaid Discount Offer")

    choice = input("\nChoose an option (1 or 2): ")

    if choice == "1":

        print(get_whatsapp_message(customer_name, order_id))

        reply = input(
            "\nEnter customer reply (YES / NO / random / leave blank): "
        )

        result = process_customer_reply(reply)

    elif choice == "2":

        print(
            get_prepaid_message(
                customer_name,
                order_id,
                order_value,
                discount
            )
        )

        reply = input(
            "\nAccept prepaid offer? (YES / NO / random / leave blank): "
        )

        result = process_prepaid_reply(
            reply,
            order_value,
            discount
        )

    else:
        print("Invalid option. Please choose 1 or 2.")
        exit()

    print("\n========== RESULT ==========")

    for key, value in result.items():
        print(f"{key}: {value}")
