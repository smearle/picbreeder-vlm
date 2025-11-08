/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.tools;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import client.utilities.ArgumentParser;

/** 
 * This program takes the delphiNEAT XML files (the spaceship ones) and converts 
 * them to a WebNEAT compatible format.
 * 
 * @author Adam Campbell
 */
public class DelphiNEATConverter {
	public static void main(String[] args){
		if(args.length == 0) {
			System.out.println("Usage: java client.tools.DelphiNEATConverter [options]");
			System.out.println("Options:");
			System.out.println("    -o outputFile");
			System.out.println("    -i inputFile");
			return;
		}

		ArgumentParser argParser = new ArgumentParser(args);
		
		String inputFile = argParser.findArgument("-i");
		String outputFile = argParser.findArgument("-o");
		String seriesId;
		
		try {
			Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(new File(inputFile));
			Document output = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
			
			Element parentElement, currentElement, genomeElement, delphiGenome = doc.getDocumentElement();
			NodeList list;
			
			seriesId = "delphiImport";
			
			parentElement = output.createElement("series");
			parentElement.setAttribute("branch", seriesId);
			output.appendChild(parentElement);
			
			currentElement = output.createElement("generation");
			currentElement.setAttribute("size", "1");
			parentElement.appendChild(currentElement);
			
			parentElement = currentElement;
			genomeElement = output.createElement("genome");
			genomeElement.setAttribute("age", "1");
			parentElement.appendChild(genomeElement);
			
			currentElement = output.createElement("identifier");
			currentElement.setAttribute("branch", seriesId);
			currentElement.setAttribute("id", "1");
			genomeElement.appendChild(currentElement);
			
			currentElement = output.createElement("parents");
			currentElement.setAttribute("count", "0");
			genomeElement.appendChild(currentElement);
			
			//if(actFunction.equals("Gausian")) actFunction = "gaussian(x)";
			//else actFunction = "sigmoid(x)";

			list = delphiGenome.getElementsByTagName("Node");
			currentElement = output.createElement("nodes");
			currentElement.setAttribute("count", ""+list.getLength());
			genomeElement.appendChild(currentElement);
			parentElement = currentElement;
			
			String[] inputs = new String[]{"BIAS_NOT_USED","x","y","d"};
			int inputIndex = 0;
			for(int i = 0; i < list.getLength(); i++){
				String id = ((Element)list.item(i)).getAttribute("ID");
				String type = ((Element)list.item(i)).getAttribute("NodeType");
				String transferFunction = ((Element)list.item(i)).getAttribute("TransferFunction");
				
				Element nodeElement = output.createElement("node");
				parentElement.appendChild(nodeElement);
				
				if(type.equals("input")){
					if(inputIndex == 0)
						nodeElement.setAttribute("label", "bias");
					else
						nodeElement.setAttribute("label", inputs[inputIndex]);
					inputIndex++;
				}else if(type.equals("output"))
					nodeElement.setAttribute("label", "ink");
				
				if(type.equals("input"))
					nodeElement.setAttribute("type", "in");
				else if(type.equals("output"))
					nodeElement.setAttribute("type", "out");
				else if(type.equals("hidden"))
					nodeElement.setAttribute("type", "hidden");
				else
					throw new RuntimeException("Unknown node type: " + type);
				
				currentElement = output.createElement("marking");
				currentElement.setAttribute("branch", seriesId);
				currentElement.setAttribute("id", ""+id);
				nodeElement.appendChild(currentElement);
				
				currentElement = output.createElement("activation");
				if(type.equals("input"))
					currentElement.setTextContent("identity(x)");
				else if(transferFunction.equals("Sigmoid"))
					currentElement.setTextContent("delphi.sigmoid(x)");
				else if(transferFunction.equals("Gaussian"))
					currentElement.setTextContent("delphi.gaussian(x)");
				nodeElement.appendChild(currentElement);
				
			}
			
			list = delphiGenome.getElementsByTagName("Link");
			currentElement = output.createElement("links");
			currentElement.setAttribute("count", ""+list.getLength());
			genomeElement.appendChild(currentElement);
			parentElement = currentElement;
			
			for(int i = 0; i < list.getLength(); i++){
				String id = ((Element)list.item(i)).getAttribute("InnovationID");
				String srcId = ((Element)((Element)list.item(i)).getElementsByTagName("TailNodeID").item(0)).getTextContent();
				String tgtId = ((Element)((Element)list.item(i)).getElementsByTagName("HeadNodeID").item(0)).getTextContent();
				String weight = ((Element)((Element)list.item(i)).getElementsByTagName("Weight").item(0)).getTextContent();
				
				Element connElement = output.createElement("link");
				parentElement.appendChild(connElement);
				
				currentElement = output.createElement("marking");
				currentElement.setAttribute("branch", seriesId);
				currentElement.setAttribute("id", ""+id);
				connElement.appendChild(currentElement);
				
				currentElement = output.createElement("source");
				currentElement.setAttribute("branch", seriesId);
				currentElement.setAttribute("id", ""+srcId);
				connElement.appendChild(currentElement);
				
				currentElement = output.createElement("target");
				currentElement.setAttribute("branch", seriesId);
				currentElement.setAttribute("id", ""+tgtId);
				connElement.appendChild(currentElement);
				
				currentElement = output.createElement("weight");
				currentElement.setTextContent(""+weight);
				connElement.appendChild(currentElement);
			}
			
			TransformerFactory transFactory = TransformerFactory.newInstance();
			transFactory.setAttribute("indent-number", 4);
			Transformer trans = transFactory.newTransformer();
			trans.setOutputProperty(OutputKeys.INDENT, "yes");
			trans.transform(new DOMSource(output), new StreamResult(new OutputStreamWriter(new FileOutputStream(outputFile))));
			
		}catch(Exception e) {
			throw new RuntimeException("XML Errors");
		}
		
	}
}
