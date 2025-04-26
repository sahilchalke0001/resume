import os
import googleapiclient.discovery
import googleapiclient.errors # Import errors for specific handling

# You might need to install the google-api-python-client library:
# pip install google-api-python-client

def get_youtube_links(query_list, api_key, max_results=3):
    """
    Searches YouTube for videos based on a list of queries and
    returns a list of dictionaries containing video details (URL, title, thumbnail).

    Args:
        query_list (list): A list of search terms.
        api_key (str): Your Google API key with access to the YouTube Data API.
        max_results (int): Max videos per query.

    Returns:
        list: A list of dictionaries, each with 'id', 'url', 'title', 'thumbnail'.
              Returns an empty list if the API key is missing or an error occurs.
    """
    videos_data = []
    processed_ids = set() # Keep track of added video IDs to avoid duplicates

    if not api_key:
        print("Warning: YouTube API key not provided. Cannot fetch video links.")
        return videos_data

    try:
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key
        )

        for query in query_list:
            print(f"Searching YouTube for: {query}")
            try:
                request = youtube.search().list(
                    q=query,
                    part="snippet",  # We need snippet for title and thumbnails
                    type="video",
                    maxResults=max_results
                )
                response = request.execute()

                for item in response.get("items", []):
                    # Ensure item is a video and has necessary details
                    if "videoId" in item.get("id", {}) and item.get("snippet"):
                        video_id = item["id"]["videoId"]

                        # Avoid duplicates if the same video matches multiple queries
                        if video_id not in processed_ids:
                            processed_ids.add(video_id)
                            title = item["snippet"]["title"]
                            # Standard YouTube watch URL
                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                            # Get medium quality thumbnail URL (good balance)
                            # Default: item['snippet']['thumbnails']['default']['url'] (120x90)
                            # Medium: item['snippet']['thumbnails']['medium']['url'] (320x180)
                            # High: item['snippet']['thumbnails']['high']['url'] (480x360)
                            thumbnail_url = item['snippet']['thumbnails'].get('high', item['snippet']['thumbnails']['default'])['url']

                            videos_data.append({
                                'id': video_id,
                                'url': video_url,
                                'title': title,
                                'thumbnail': thumbnail_url
                            })
                    else:
                        # Log if an item is skipped (optional)
                        # print(f"Skipping non-video item or item without snippet: {item.get('id')}")
                        pass

            except googleapiclient.errors.HttpError as e:
                print(f"An HTTP error occurred during YouTube API call for query '{query}': {e}")
                # Decide if you want to stop or continue with the next query
                # continue
            except Exception as e:
                 print(f"An unexpected error occurred processing query '{query}': {e}")
                 # continue


    except googleapiclient.errors.HttpError as e:
         print(f"A general HTTP error occurred initializing YouTube service or during API calls: {e}")
         return [] # Can't proceed without the service
    except Exception as e:
        print(f"An unexpected error occurred during Youtube setup: {e}")
        return [] # Return empty list on major failure


    print(f"Found {len(videos_data)} unique YouTube video details.")
    return videos_data

# Example usage (ensure your backend uses this function and passes its result)
# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     load_dotenv()
#     YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
#     if YOUTUBE_API_KEY:
#         test_queries = ["resume writing tips", "job interview skills"]
#         videos = get_youtube_links(test_queries, YOUTUBE_API_KEY, max_results=3)
#         if videos:
#             print("\nFound Videos:")
#             for video in videos:
#                 print(f"  Title: {video['title']}")
#                 print(f"  URL: {video['url']}")
#                 print(f"  Thumbnail: {video['thumbnail']}\n")
#         else:
#             print("\nNo video details found or an error occurred.")
#     else:
#         print("YOUTUBE_API_KEY not set in .env file. Cannot test Youtube.")